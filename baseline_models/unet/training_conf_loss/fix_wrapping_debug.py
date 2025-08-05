import torch
import modulus
import numpy as np
import xarray as xr
from wrap_model_debug import WrappedModel
import os, gc
import sys
sys.path.append('/path/to/directory/containing/climsim_utils')
from climsim_utils.data_utils import *

import argparse

parser = argparse.ArgumentParser(description="Load save path")
parser.add_argument('save_path', type=str, help='Path for saved model')
args = parser.parse_args()
save_path = args.save_path

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

input_sub, input_div, out_scale = data.save_norm(write=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_file = os.path.join(save_path, 'model.mdlus')

# convert the model to torchscript
device = torch.device("cpu")
model_inf = modulus.Module.from_checkpoint(save_file).to(device)
# scripted_model = torch.jit.script(model_inf)
# scripted_model = scripted_model.eval()
# save_file_torch = os.path.join(save_path, 'model.pt')
# scripted_model.save(save_file_torch)

# wrap model
device = torch.device("cuda")
wrapped_model = WrappedModel(original_model = model_inf,
                                input_sub = torch.tensor(input_sub, dtype=torch.float32).to(device),
                                input_div = torch.tensor(input_div, dtype=torch.float32).to(device),
                                out_scale = torch.tensor(out_scale, dtype=torch.float32).to(device),
                                qn_lbd = torch.tensor(qn_lbd, dtype=torch.float32).to(device)).to(device)
save_file_wrapped = os.path.join(save_path, 'wrapped_model_debug.pt')
scripted_model_wrapped = torch.jit.script(wrapped_model)
scripted_model_wrapped = scripted_model_wrapped.eval()
scripted_model_wrapped.save(save_file_wrapped)

# mdlus_directory = os.path.join(save_path, 'ckpt')
# wrapped_directory = os.path.join(save_path, 'wrapped')
# for filename in os.listdir(mdlus_directory):
#     print(filename)
#     if filename.endswith(".mdlus"):
#         full_path = os.path.join(mdlus_directory, filename)
#         print(full_path)
#         model_inf = modulus.Module.from_checkpoint(full_path).to(device)
#         scripted_model = torch.jit.script(model_inf)
#         scripted_model = scripted_model.eval()

#         # Save the TorchScript model
#         save_path_torch = os.path.join(mdlus_directory, filename.replace('.mdlus', '.pt'))
#         scripted_model.save(save_path_torch)
#         print('save path for ckpt torchscript:', save_path_torch)
        
#         # wrap model
#         device = torch.device("cuda")
#         wrapped_model = WrappedModel(original_model = model_inf,
#                                     input_sub = torch.tensor(input_sub, dtype=torch.float32).to(device),
#                                     input_div = torch.tensor(input_div, dtype=torch.float32).to(device),
#                                     out_scale = torch.tensor(out_scale, dtype=torch.float32).to(device),
#                                     qn_lbd = torch.tensor(qn_lbd, dtype=torch.float32).to(device)).to(device)
#         save_path_wrapped = os.path.join(wrapped_directory, filename.replace('.mdlus', '_wrapped.pt'))
#         scripted_model_wrapped = torch.jit.script(wrapped_model)
#         scripted_model_wrapped = scripted_model_wrapped.eval()
#         scripted_model_wrapped.save(save_path_wrapped)