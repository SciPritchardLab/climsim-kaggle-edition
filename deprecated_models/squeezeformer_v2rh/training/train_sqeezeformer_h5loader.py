import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pickle
from dataclasses import dataclass
import modulus
from modulus.metrics.general.mse import mse
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
from climsim_datapip import climsim_dataset
from climsim_datapip_h5 import climsim_dataset_h5
from sqeezeformer import Sqeezeformer
import sqeezeformer as sqeezeformer

import hydra
from collections.abc import Iterable
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()

    # grid_path = cfg.climsim_path+'/grid_info/ClimSim_low-res_grid-info.nc'
    # norm_path = cfg.climsim_path+'/preprocessing/normalizations/'


    if cfg.variable_subsets == 'v2_rh':
        input_size = 557 - 1 # exclude cam_in_SNOWHICE as kaggle data does not have it
        output_size = 368

    # get scaling arrays, following GreySnow's solution on kaggle
    data_path_MSM = '/global/homes/z/zeyuanhu/scratch/kaggle_datasets/'
    min_y = pickle.load((open(f'{data_path_MSM}/min_y.p', 'br')))
    max_y = pickle.load((open(f'{data_path_MSM}/max_y.p', 'br')))
    mean_y = pickle.load((open(f'{data_path_MSM}/mean_y.p', 'br')))
    std_y = pickle.load((open(f'{data_path_MSM}/std_y.p', 'br')))

    min_col_not = pickle.load((open(f'{data_path_MSM}/min_col_not.p', 'br')))
    max_col_not = pickle.load((open(f'{data_path_MSM}/max_col_not.p', 'br')))
    mean_col_not = pickle.load((open(f'{data_path_MSM}/mean_col_not.p', 'br')))
    std_col_not = pickle.load((open(f'{data_path_MSM}/std_col_not.p', 'br')))

    min_col_not_test = pickle.load((open(f'{data_path_MSM}/min_col_not_test.p', 'br')))
    max_col_not_test = pickle.load((open(f'{data_path_MSM}/max_col_not_test.p', 'br')))
    mean_col_not_test = pickle.load((open(f'{data_path_MSM}/mean_col_not_test.p', 'br')))
    std_col_not_test = pickle.load((open(f'{data_path_MSM}/std_col_not_test.p', 'br')))

    x_col_min = pickle.load((open(f'{data_path_MSM}/x_col_min.p', 'br')))
    x_col_max = pickle.load((open(f'{data_path_MSM}/x_col_max.p', 'br')))
    x_col_mean = pickle.load((open(f'{data_path_MSM}/x_col_mean.p', 'br')))
    x_col_std = pickle.load((open(f'{data_path_MSM}/x_col_std.p', 'br')))

    x_col_min_test = pickle.load((open(f'{data_path_MSM}/x_col_min_test.p', 'br')))
    x_col_max_test = pickle.load((open(f'{data_path_MSM}/x_col_max_test.p', 'br')))
    x_col_mean_test = pickle.load((open(f'{data_path_MSM}/x_col_mean_test.p', 'br')))
    x_col_std_test = pickle.load((open(f'{data_path_MSM}/x_col_std_test.p', 'br')))

    X_total_min = pickle.load((open(f'{data_path_MSM}/X_total_min.p', 'br')))
    X_total_max = pickle.load((open(f'{data_path_MSM}/X_total_max.p', 'br')))
    X_total_mean = pickle.load((open(f'{data_path_MSM}/x_total_mean.p', 'br')))
    X_total_std = pickle.load((open(f'{data_path_MSM}/x_total_std.p', 'br')))

    X_total_min_test = pickle.load((open(f'{data_path_MSM}/X_total_min_test.p', 'br')))
    X_total_max_test = pickle.load((open(f'{data_path_MSM}/X_total_max_test.p', 'br')))
    X_total_mean_test = pickle.load((open(f'{data_path_MSM}/x_total_mean_test.p', 'br')))
    X_total_std_test = pickle.load((open(f'{data_path_MSM}/x_total_std_test.p', 'br')))

    y_total_min = pickle.load((open(f'{data_path_MSM}/y_total_min.p', 'br')))
    y_total_max = pickle.load((open(f'{data_path_MSM}/y_total_max.p', 'br')))
    y_total_mean = pickle.load((open(f'{data_path_MSM}/y_total_mean.p', 'br')))
    y_total_std = pickle.load((open(f'{data_path_MSM}/y_total_std.p', 'br')))

    stds_new = pickle.load(open(f'{data_path_MSM}/stds/stds_new.p', 'br')) # mask for loss, 0 for features that are not used in loss
    stds_new[120+12:120+26] = 0
    loss_mask_tensor = torch.tensor(stds_new, dtype=torch.float32).unsqueeze(0)


    stds_temp = np.where(std_y == 0, 0, 1/np.where(std_y>0, std_y, 1))
    stds = stds_temp[0]
    mean_col_not = mean_col_not[0]
    std_col_not = std_col_not[0]
    min_col_not_norm = ((np.where(min_col_not<min_col_not_test, min_col_not , min_col_not_test)-mean_col_not)/std_col_not)[0]
    X_total_mean = X_total_mean[0]
    X_total_std = X_total_std[0]
    X_total_norm_min = ((np.where(X_total_min<X_total_min_test, X_total_min , X_total_min_test)-X_total_mean)/X_total_std)[0]

    x_col_mean = x_col_mean[0]
    x_col_std = x_col_std[0]
    x_col_norm_min = ((np.where(x_col_min<x_col_min_test, x_col_min , x_col_min_test)-x_col_mean)/x_col_std)[0]

    mean_y = mean_y[0]
    x_col_min = x_col_min[0]

    X_total_mean[1,0] = 0
    x_col_mean[60:120] = 0
    X_total_std[1,0] = 1
    x_col_std[60:120] = 1

    x_col_std[540-33:540] = 0
    x_col_std[480-33:480] = 0
    x_col_norm_min[540-33:540] = 0
    x_col_norm_min[480-33:480] = 0

    norm_arrays = {'input_size': input_size,
                    'output_size': output_size,
                   'mean_y': mean_y,
                   'stds': stds,
                   'mean_col_not': mean_col_not,
                   'std_col_not': std_col_not,
                   'X_total_mean': X_total_mean,
                   'X_total_std': X_total_std,
                   'x_col_mean': x_col_mean,
                   'x_col_std': x_col_std,
                   'x_col_norm_min': x_col_norm_min,
                   }

    val_input_path = cfg.data_path + cfg.val_input
    val_target_path = cfg.data_path + cfg.val_target
    if not os.path.exists(cfg.data_path + cfg.val_input):
        raise ValueError('Validation input path does not exist')

    val_dataset = climsim_dataset(val_input_path, val_target_path, norm_arrays)

    #train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    train_dataset = climsim_dataset_h5(cfg.data_path, norm_arrays)
            
    train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False if dist.distributed else True,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=cfg.num_workers)

    

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print('MLP init arguments: ', input_size, output_size, tmp_mlp_hidden_dims, tmp_mlp_layers, tmp_output_prune, tmp_strato_lev)
    model = Sqeezeformer(
        # col_len = (9*2+2)*60, 
        col_len = (9*2+2)*60, 
        col_not_len = 16,
        dim=cfg.squeeze_dim, 
        head_dim=cfg.squeeze_head_dim,
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
                find_unused_parameters=True, #dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError('Optimizer not implemented')
    
    # create scheduler
    if cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    elif cfg.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
    elif cfg.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
    elif cfg.scheduler_name == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.cosine_warmup.T_0, T_mult=cfg.scheduler.cosine_warmup.T_mult, eta_min=cfg.scheduler.cosine_warmup.eta_min)
    else:
        raise ValueError('Scheduler not implemented')
    
    # create loss function
    if cfg.loss == 'mse':
        loss_fn = mse
        criterion = nn.MSELoss()
    elif cfg.loss == 'mae':
        loss_fn = nn.L1Loss()
        criterion = nn.L1Loss()
    elif cfg.loss == 'huber':
        loss_fn = nn.SmoothL1Loss()
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError('Loss function not implemented')
    
    
    # Initialize the console logger
    logger = PythonLogger("main")  # General python logger

    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="zeyuan_hu",
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
        logger.info("Checkpoints should be set >0, setting to 1")
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
      

    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        cuda_graph_warmup=13,
    )
    def training_step(model, data_input, target):
        output = model(data_input)
        # output = output * loss_mask_tensor
        # target = target * loss_mask_tensor
        preds = output[:,:368]
        confidence = output[:,368:]
        
        # loss = criterion(output, target)
        loss = torch.abs(target - preds)
        loss = loss * loss_mask_tensor

        loss2 = torch.abs(loss - confidence)
        loss2 = loss2 * loss_mask_tensor
        loss = torch.mean(loss + loss2)
        return loss
    
    @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)
    #training block
    loss_mask_tensor = loss_mask_tensor.to(device)
    logger.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(cfg.epochs):
        if dist.distributed:
            train_sampler.set_epoch(epoch)
        
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=100) as launchlog:
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

                preds = output[:,:368]
                confidence = output[:,368:]
                loss = torch.abs(target - preds)
                loss = loss * loss_mask_tensor
                mae_loss = torch.mean(loss)
                loss2 = torch.abs(loss - confidence)
                loss2 = loss2 * loss_mask_tensor
                loss = torch.mean(loss + loss2)

                loss.backward()
                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    scheduler.step(epoch + iteration / total_iterations)

                # loss = training_step(model, data_input, target)
                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"], "loss_train_mae": mae_loss.detach().cpu().numpy()})
                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch+1}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1


            
            
            model.eval()
            val_loss = 0.0
            val_loss_mae = 0.0
            num_samples_processed = 0
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/1 [Validation]')
            current_step = 0
            for data_input, target in val_loop:
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                
                data_input, target = data_input.to(device), target.to(device)

                output = eval_step_forward(model, data_input)
                # output = output * loss_mask_tensor
                # target = target * loss_mask_tensor
                # loss = criterion(output, target)
                preds = output[:,:368]
                confidence = output[:,368:]
                
                # loss = criterion(output, target)
                loss = torch.abs(target - preds)
                loss = loss * loss_mask_tensor
                mae_loss = torch.mean(loss)
                loss2 = torch.abs(loss - confidence)
                loss2 = loss2 * loss_mask_tensor
                loss = torch.mean(loss + loss2)

                val_loss += loss.item() * data_input.size(0)
                val_loss_mae += mae_loss.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)
                current_val_loss_avg_mae = val_loss_mae / num_samples_processed
                current_step += 1
                del data_input, target, output
                    
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size

                current_val_loss_avg_mae = torch.tensor(current_val_loss_avg_mae, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg_mae)
                current_val_loss_avg_mae = current_val_loss_avg_mae.item() / dist.world_size

            if dist.rank == 0:
                launchlog.log_epoch({"loss_valid": current_val_loss_avg, "loss_valid_mae": current_val_loss_avg_mae})

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
                pass # handled in the optimizer.step() in training loop
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
        sqeezeformer.device = "cpu"
        device = torch.device("cpu")
        model_inf = modulus.Module.from_checkpoint(save_file).to(device)
        scripted_model = torch.jit.script(model_inf)
        scripted_model = scripted_model.eval()
        save_file_torch = os.path.join(save_path, 'model.pt')
        scripted_model.save(save_file_torch)
        # # save input and output normalizations
        # data.save_norm(save_path, True)
        logger.info("saved input/output normalizations and model to: " + save_path)

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