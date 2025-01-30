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
from paoModel import paoModel
import transformers

import hydra
from collections.abc import Iterable
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc
from soap import SOAP
from torch.nn.utils import clip_grad_norm_

# config_name gets overwritten in the SLURM script
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

    model = paoModel().to(dist.device)