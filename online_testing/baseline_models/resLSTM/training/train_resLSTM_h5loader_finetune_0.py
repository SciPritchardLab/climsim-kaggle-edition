import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
import modulus
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from climsim_utils.data_utils import *
from dataset_val import dataset_val
from dataset_train import dataset_train
from resLSTM import resLSTM

import hydra
from collections.abc import Iterable
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc
from torch.nn.utils import clip_grad_norm_

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()

    grid_info = xr.open_dataset(cfg.grid_info)
    input_mean = xr.open_dataset(cfg.input_mean)
    input_max = xr.open_dataset(cfg.input_max)
    input_min = xr.open_dataset(cfg.input_min)
    output_scale = xr.open_dataset(cfg.output_scale)

    lbd_qn = np.loadtxt(cfg.qn_lbd, delimiter=',')

    data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale)

    # set variables to subset
    if cfg.variable_subsets == 'v1': 
        data.set_to_v1_vars()
    elif cfg.variable_subsets == 'v1_dyn':
        data.set_to_v1_dyn_vars()
    elif cfg.variable_subsets == 'v2':
        data.set_to_v2_vars()
    elif cfg.variable_subsets == 'v2_dyn':
        data.set_to_v2_dyn_vars()
    elif cfg.variable_subsets == 'v2_rh':
        data.set_to_v2_rh_vars()
    elif cfg.variable_subsets == 'v3':
        data.set_to_v3_vars()
    elif cfg.variable_subsets == 'v4':
        data.set_to_v4_vars()
    elif cfg.variable_subsets == 'v5':
        data.set_to_v5_vars()
    elif cfg.variable_subsets == 'v6':
        data.set_to_v6_vars()
    else:
        raise ValueError('Unknown variable subset')

    input_size = data.input_feature_len
    output_size = data.target_feature_len

    input_sub, input_div, out_scale = data.save_norm(save_path = '.', write=True)

    val_input_path = cfg.val_input
    val_target_path = cfg.val_target
    if not os.path.exists(cfg.val_input):
        raise ValueError('Validation input path does not exist')

    val_dataset = dataset_val(input_paths = val_input_path, 
                                  target_paths = val_target_path, 
                                  input_sub = input_sub, 
                                  input_div = input_div, 
                                  out_scale = out_scale, 
                                  qinput_prune = cfg.qinput_prune, 
                                  output_prune = cfg.output_prune, 
                                  strato_lev = cfg.strato_lev, 
                                  strato_lev_out = cfg.strato_lev_out, 
                                  qn_lbd = lbd_qn, 
                                  decouple_cloud = cfg.decouple_cloud, 
                                  aggressive_pruning = cfg.aggressive_pruning, 
                                  strato_lev_qinput = cfg.strato_lev_qinput, 
                                  strato_lev_tinput = cfg.strato_lev_tinput, 
                                  input_clip = cfg.input_clip, 
                                  input_clip_rhonly = cfg.input_clip_rhonly)

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    train_dataset = dataset_train(parent_path = cfg.data_path, 
                                    input_sub = input_sub, 
                                    input_div = input_div, 
                                    out_scale = out_scale, 
                                    qinput_prune = cfg.qinput_prune, 
                                    output_prune = cfg.output_prune, 
                                    strato_lev = cfg.strato_lev, 
                                    strato_lev_out = cfg.strato_lev_out, 
                                    qn_lbd = lbd_qn, 
                                    decouple_cloud = cfg.decouple_cloud, 
                                    aggressive_pruning = cfg.aggressive_pruning, 
                                    strato_lev_qinput = cfg.strato_lev_qinput, 
                                    strato_lev_tinput = cfg.strato_lev_tinput, 
                                    input_clip = cfg.input_clip, 
                                    input_clip_rhonly = cfg.input_clip_rhonly)
            
    train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False if dist.distributed else True,
                                sampler=train_sampler,
                                pin_memory=True,
                                drop_last=True,
                                num_workers=cfg.num_workers)
                              
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print('debug: output_size', output_size, output_size//60, output_size%60)
    model = resLSTM(
        inputs_dim = 42,
        num_lstm = 10,
        hidden_state = 512,
    ).to(dist.device)

    if len(cfg.restart_path) > 0:
        print("Restarting from checkpoint: " + cfg.restart_path)
        if dist.distributed:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            if dist.rank == 0:
                model.load_state_dict(model_restart.state_dict())
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                model.load_state_dict(model_restart.state_dict())
        else:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            model.load_state_dict(model_restart.state_dict())

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # create optimizer
    optimizer = None
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    if optimizer is None:
        raise ValueError('Optimizer not implemented')
    
    # create scheduler
    scheduler = None
    if cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    elif cfg.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
    elif cfg.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
    elif cfg.scheduler_name == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.cosine_warmup.T_0, T_mult=cfg.scheduler.cosine_warmup.T_mult, eta_min=cfg.scheduler.cosine_warmup.eta_min)
    if scheduler is None:
        raise ValueError('Scheduler not implemented')
    
    # create loss function
    criterion = None
    if cfg.loss == 'mse':
        loss_fn = mse
        criterion = nn.MSELoss()
    elif cfg.loss == 'mae':
        loss_fn = nn.L1Loss()
        criterion = nn.L1Loss()
    elif cfg.loss == 'smoothL1':
        loss_fn = nn.SmoothL1Loss()
        criterion = nn.SmoothL1Loss()
    if criterion is None:
        raise ValueError('Loss function not implemented')
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    def consistency_loss(pred, target):

        # pred should be of shape (batch_size, 368)
        # target should be of shape (batch_size, 368)
        # 0-60: dt, 60-120 dq1, 120-180 dq2, 180-240 dq3, 240-300 du, 300-360 dv, 360-368 d2d
        def vert_diff(col):
            return col[:,1:60] - col[:,0:59]
        custom_loss = criterion(pred[:,0:60], target[:,0:60])
        custom_loss += criterion(vert_diff(pred[:,0:60]), vert_diff(target[:,0:60]))
        # custom_loss += criterion(vert_diff(pred[:,60:120]), vert_diff(target[:,60:120]))
        # custom_loss += criterion(vert_diff(pred[:,120:180]), vert_diff(target[:,120:180]))
        # custom_loss += criterion(vert_diff(pred[:,180:240]), vert_diff(target[:,180:240]))
        # custom_loss += criterion(vert_diff(pred[:,240:300]), vert_diff(target[:,240:300]))
        return custom_loss
    
    # Initialize the console logger
    logger = PythonLogger("main")  # General python logger

    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="cbrain",
            mode="online",
        )
        LaunchLogger.initialize(use_wandb=True)
    else:
        # Initialize the MLFlow logger
        initialize_mlflow(
            experiment_name=cfg.mlflow.project,
            experiment_desc="Modulus launch development",
            run_name=cfg.expname,
            run_desc="Modulus Training",
            user_name="Modulus User",
            mode="offline",
        )
        LaunchLogger.initialize(use_mlflow=True)

    if cfg.save_top_ckpts<=0:
        logger.info("Checkpoints should be set > 0, setting to 1")
        num_top_ckpts = 1
    else:
        num_top_ckpts = cfg.save_top_ckpts

    if cfg.top_ckpt_mode == 'min':
        top_checkpoints = [(float('inf'), None)] * num_top_ckpts
    elif cfg.top_ckpt_mode == 'max':
        top_checkpoints = [(-float('inf'), None)] * num_top_ckpts
    else:
        raise ValueError('Unknown top_ckpt_mode')
    
    if dist.rank == 0:
        save_path = os.path.join(cfg.save_path, cfg.expname) #cfg.save_path + cfg.expname
        save_path_ckpt = os.path.join(save_path, 'ckpt')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
    
    if dist.world_size > 1:
        torch.distributed.barrier()

    @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)
    #training block
    logger.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(cfg.epochs):
        if dist.distributed:
            train_sampler.set_epoch(epoch)

        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=cfg.mini_batch_log_freq) as launchlog:
            model.train()
            
            total_iterations = len(train_loader)

            train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            current_step = 0
            for iteration, (data_input, target) in enumerate(train_loop):
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                data_input, target = data_input.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data_input)
                loss = consistency_loss(output, target)
                loss.backward()

                if cfg.clip_grad:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    scheduler.step(epoch + iteration / total_iterations)

                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"]})
                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch+1}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1
            
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_mae = 0.0
            num_samples_processed = 0
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/1 [Validation]')
            current_step = 0
            for data_input, target in val_loop:
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break

                data_input, target = data_input.to(device), target.to(device)

                output = eval_step_forward(model, data_input)
                loss = consistency_loss(output, target)
                mse = mse_criterion(output, target)
                mae = mae_criterion(output, target)
                val_loss += loss.item() * data_input.size(0)
                val_mse += mse.item() * data_input.size(0)
                val_mae += mae.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                current_val_mse_avg = val_mse / num_samples_processed
                current_val_mae_avg = val_mae / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)
                current_step += 1
                del data_input, target, output
                    
            
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size

                current_val_mse_avg = torch.tensor(current_val_mse_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_mse_avg)
                current_val_mse_avg = current_val_mse_avg.item() / dist.world_size

                current_val_mae_avg = torch.tensor(current_val_mae_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_mae_avg)
                current_val_mae_avg = current_val_mae_avg.item() / dist.world_size

            if dist.rank == 0:
                launchlog.log_epoch({"loss_valid": current_val_loss_avg})
                launchlog.log_epoch({"mse_valid": current_val_mse_avg})
                launchlog.log_epoch({"mae_valid": current_val_mae_avg})
                current_metric = current_val_loss_avg
                # Save the top checkpoints
                if cfg.top_ckpt_mode == 'min':
                    is_better = current_metric < max(top_checkpoints, key=lambda x: x[0])[0]
                elif cfg.top_ckpt_mode == 'max':
                    is_better = current_metric > min(top_checkpoints, key=lambda x: x[0])[0]
                
                #print('debug: is_better', is_better, current_metric, top_checkpoints)
                if len(top_checkpoints) == 0 or is_better:
                    ckpt_path = os.path.join(save_path_ckpt, f'ckpt_epoch_{epoch+1}_metric_{current_metric:.4f}.mdlus')
                    if dist.distributed:
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)
                    top_checkpoints.append((current_metric, ckpt_path))
                    # Sort and keep top 5 based on max/min goal at the beginning
                    if cfg.top_ckpt_mode == 'min':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=False)
                    elif cfg.top_ckpt_mode == 'max':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    # delete the worst checkpoint
                    if len(top_checkpoints) > num_top_ckpts:
                        worst_ckpt = top_checkpoints.pop()
                        print(f"Removing worst checkpoint: {worst_ckpt[1]}")
                        if worst_ckpt[1] is not None:
                            os.remove(worst_ckpt[1])
                            
            if cfg.scheduler_name == 'plateau':
                scheduler.step(current_val_loss_avg)
            elif cfg.scheduler_name == 'cosine_warmup':
                pass # handled in optimizer.step() in training loop
            else:
                scheduler.step()
            
            if dist.world_size > 1:
                torch.distributed.barrier()
                
    if dist.rank == 0:
        logger.info("Start recovering the model from the top checkpoint to do torchscript conversion")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)
        # convert the model to torchscript
        device = torch.device("cpu")
        model_inf = modulus.Module.from_checkpoint(save_file).to(device)
        scripted_model = torch.jit.script(model_inf)
        scripted_model = scripted_model.eval()
        save_file_torch = os.path.join(save_path, 'model.pt')
        scripted_model.save(save_file_torch)
        logger.info(f"saved input/output normalizations and model to: {save_path}")

        mdlus_directory = os.path.join(save_path, 'ckpt')
        for filename in os.listdir(mdlus_directory):
            print(filename)
            if filename.endswith(".mdlus"):
                full_path = os.path.join(mdlus_directory, filename)
                print(full_path)
                model = modulus.Module.from_checkpoint(full_path).to("cpu")
                scripted_model = torch.jit.script(model)
                scripted_model = scripted_model.eval()

                # Save the TorchScript model
                save_path_torch = os.path.join(mdlus_directory, filename.replace('.mdlus', '.pt'))
                scripted_model.save(save_path_torch)
                print('save path for ckpt torchscript:', save_path_torch)


        logger.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()