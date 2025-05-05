import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
import modulus
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from omegaconf import OmegaConf
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from climsim_utils.data_utils import *

from climsim_datasets import TrainingDataset, ValidationDataset
from squeezeformer import Squeezeformer
from wrap_model import WrappedModel
import hydra
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import os, gc
import random

grid_info = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc')
input_mean = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_mean_v2_rh_mc_pervar.nc')
input_max = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_max_v2_rh_mc_pervar.nc')
input_min = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_min_v2_rh_mc_pervar.nc')
output_scale = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/output_scale_std_lowerthred_v2_rh_mc.nc')
qn_lbd = np.loadtxt('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/qn_exp_lambda_large.txt', delimiter=',')

data = data_utils(grid_info = grid_info, 
                    input_mean = input_mean, 
                    input_max = input_max, 
                    input_min = input_min, 
                    output_scale = output_scale)

data.set_to_v2_rh_mc_vars()

input_size = data.input_feature_len
output_size = data.target_feature_len
per_lev_sub = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_sub.nc')
per_lev_div = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_div.nc')
per_col_sub = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_col_sub.nc')
per_col_div = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_col_div.nc')
per_lev_min_norm = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_min_norm.nc')

input_sub_per_lev, input_div_per_lev, input_sub_per_col, input_div_per_col, input_min_norm_per_lev, out_scale = data.save_norm_multirep(per_lev_sub,
                                                                                                                                        per_lev_div,
                                                                                                                                        per_col_sub,
                                                                                                                                        per_col_div,
                                                                                                                                        per_lev_min_norm,
                                                                                                                                        save_path = '',                    
                                                                                                                                        write=False)

train_dataset = TrainingDataset(parent_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/train_set/',
                                input_sub_per_lev = input_sub_per_lev,
                                input_div_per_lev = input_div_per_lev,
                                input_sub_per_col = input_sub_per_col,
                                input_div_per_col = input_div_per_col,
                                input_min_norm_per_lev = input_min_norm_per_lev,
                                out_scale = out_scale,
                                output_prune = True,
                                strato_lev = 15,
                                strato_lev_out = 12)

val_dataset = ValidationDataset(val_input_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_input.npy',
                                val_target_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_target.npy',
                                input_sub_per_lev = input_sub_per_lev,
                                input_div_per_lev = input_div_per_lev,
                                input_sub_per_col = input_sub_per_col,
                                input_div_per_col = input_div_per_col,
                                input_min_norm_per_lev = input_min_norm_per_lev,
                                out_scale = out_scale,
                                output_prune = True,
                                strato_lev = 15,
                                strato_lev_out = 12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Squeezeformer(
            input_profile_num = 25,
            input_scalar_num = data.input_scalar_num,
            target_profile_num = data.target_profile_num,
            target_scalar_num = data.target_scalar_num,
            output_prune = True,
            strato_lev_out = 12,
            loc_embedding = False,
            embedding_type = 'positional',
        ).to('cuda')

criterion = nn.HuberLoss()

# @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
# def eval_step_forward(my_model, invar):
#     return my_model(invar)

val_loader = DataLoader(val_dataset, 
                        batch_size=1024, 
                        shuffle=False,
                        sampler=None,
                        num_workers=32)

model.eval()
val_loss = 0.0
num_samples_processed = 0
val_loop = tqdm(val_loader, desc=f'Epoch 1/1 [Validation]')
current_step = 0
for data_input, target in val_loop:
    print(f'Current step: {current_step}')
    # if cfg.output_prune:
    #     # the following code only works for the v2/v3 output cases!
    #     target[:,60:60+cfg.strato_lev] = 0
    #     target[:,120:120+cfg.strato_lev] = 0
    #     target[:,180:180+cfg.strato_lev] = 0
    # Move data to the device
    data_input, target = data_input.to(device), target.to(device)
    if torch.isnan(data_input).any() or torch.isinf(data_input).any():
        print(f"NaN or Inf detected in input at step {current_step}")
        print(f"Saving problematic batch at step {current_step}")
        torch.save(data_input, f"problematic_input_step_{current_step}.pt")
        break
    if torch.isnan(target).any() or torch.isinf(target).any():
        print(f"NaN or Inf detected in target at step {current_step}")
        print(f"Saving problematic batch at step {current_step}")
        torch.save(target, f"problematic_target_step_{current_step}.pt")
        break
    with torch.no_grad():
        output = model(data_input)
    # output = eval_step_forward(model, data_input)
    loss = criterion(output, target)
    if loss != loss:
        print(f'Loss is NaN at step {current_step}')
        print(f"Saving problematic batch at step {current_step}")
        torch.save(data_input, f"problematic_input_step_{current_step}.pt")
        torch.save(output, f"problematic_output_step_{current_step}.pt")
        torch.save(target, f"problematic_target_step_{current_step}.pt")
        break
    val_loss += loss.item() * data_input.size(0)
    if val_loss != val_loss:
        print(f'Val Loss is NaN at step {current_step}')
        print(f"Saving problematic batch at step {current_step}")
        torch.save(data_input, f"problematic_input_step_{current_step}.pt")
        torch.save(output, f"problematic_output_step_{current_step}.pt")
        torch.save(target, f"problematic_target_step_{current_step}.pt")
        break
    num_samples_processed += data_input.size(0)

    # Calculate and update the current average loss
    current_val_loss_avg = val_loss / num_samples_processed
    if current_val_loss_avg != current_val_loss_avg:
        print(f'Current Val Loss is NaN at step {current_step}')
        break
    val_loop.set_postfix(loss=current_val_loss_avg)
    current_step += 1
    del data_input, target, output