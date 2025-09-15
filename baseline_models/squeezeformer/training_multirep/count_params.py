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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####
# CHANGE THIS SECTION
model = Squeezeformer(
            input_profile_num = 25,
            input_scalar_num = data.input_scalar_num,
            target_profile_num = data.target_profile_num,
            target_scalar_num = data.target_scalar_num,
            output_prune = True,
            strato_lev_out = 12,
            loc_embedding = False,
            embedding_type = "positional",
        ).to(device)
# CHANGE THIS SECTION
####

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(trainable_params)