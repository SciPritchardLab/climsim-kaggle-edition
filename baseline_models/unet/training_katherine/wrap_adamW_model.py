from climsim_utils.data_utils import *

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import modulus

from climsim_unet_conf import ClimsimUnetConf
import climsim_unet_conf as climsim_unet_conf

input_mean_file = 'input_mean_v6_pervar.nc'
input_max_file = 'input_max_v6_pervar.nc'
input_min_file = 'input_min_v6_pervar.nc'
output_scale_file = 'output_scale_std_lowerthred_v6.nc'
lbd_qn_file = 'qn_exp_lambda_large.txt'

grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

f_torch_model = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adamW/model.mdlus'
save_file_torch = os.path.join('/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adamW/', 'wrapped_model.pt')

norm_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/'
input_mean_file = 'inputs/input_mean_v6_pervar.nc'
input_max_file = 'inputs/input_max_v6_pervar.nc'
input_min_file = 'inputs/input_min_v6_pervar.nc'
output_scale_file = 'outputs/output_scale_std_lowerthred_v6.nc'
lbd_qn_file = 'inputs/qn_exp_lambda_large.txt'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + input_mean_file)
input_max = xr.open_dataset(norm_path + input_max_file)
input_min = xr.open_dataset(norm_path + input_min_file)
output_scale = xr.open_dataset(norm_path + output_scale_file)
lbd_qn = np.loadtxt(norm_path + lbd_qn_file, delimiter = ',')

data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale,
                  qinput_log=False,
                  normalize=False)

data.set_to_v2_rh_mc_vars()

input_sub, input_div, out_scale = data.save_norm(write = False)

class WrappedModel(nn.Module):
    def __init__(self, original_model, input_sub, input_div, out_scale, lbd_qn):
        super(WrappedModel, self).__init__()
        self.original_model = original_model
        self.input_sub = torch.tensor(input_sub, dtype=torch.float32, device = torch.device('cuda'))
        self.input_div = torch.tensor(input_div, dtype=torch.float32, device = torch.device('cuda'))
        self.out_scale = torch.tensor(out_scale, dtype=torch.float32, device = torch.device('cuda'))
        self.lbd_qn = torch.tensor(lbd_qn, dtype=torch.float32, device = torch.device('cuda'))

    def to(self, device):
        """Ensure all tensors are moved to the correct device"""
        self.input_sub = self.input_sub.to(device)
        self.input_div = self.input_div.to(device)
        self.out_scale = self.out_scale.to(device)
        self.lbd_qn = self.lbd_qn.to(device)
        return super().to(device)
    
    def apply_temperature_rules(self, T):
        # Create an output tensor, initialized to zero
        output = torch.zeros_like(T)

        # Apply the linear transition within the range 253.16 to 273.16
        mask = (T >= 253.16) & (T <= 273.16)
        output[mask] = (T[mask] - 253.16) / 20.0  # 20.0 is the range (273.16 - 253.16)

        # Values where T > 273.16 set to 1
        output[T > 273.16] = 1

        # Values where T < 253.16 are already set to 0 by the initialization
        return output

    def preprocessing(self, x):
        # convert v2 input array to v2_rh_mc input array:
        xout = x
        xout_new = torch.zeros((xout.shape[0], 557), dtype=xout.dtype, device = x.device)
        xout_new[:,0:120] = xout[:,0:120] # state_t, state_rh
        xout_new[:,120:180] = xout[:,120:180] + xout[:,180:240] # state_qn
        xout_new[:,180:240] = self.apply_temperature_rules(xout[:,0:60]) # liq_partition
        xout_new[:,240:557] = xout[:,240:557] # state_u, state_v
        x = xout_new
        
        #do input normalization
        x[:,120:180] = 1 - torch.exp(-x[:,120:180] * self.lbd_qn.to(x.device))
        x = (x - self.input_sub.to(x.device)) / self.input_div.to(x.device)
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[:,120:120+15] = 0
        #clip rh input
        x[:, 60:120] = torch.clamp(x[:, 60:120], 0, 1.2)

        return x

    def postprocessing(self, x):
        x[:,60:75] = 0
        x[:,120:135] = 0
        x[:,180:195] = 0
        x[:,240:255] = 0
        x = x/self.out_scale
        return x

    def forward(self, x):
        #print(f"Model forward pass running on device: {x.device}")
        # Print the number of available CUDA devices
        num_gpus = torch.cuda.device_count()
        #print(f"Number of available CUDA devices: {num_gpus}")
        t_before = x[:,0:60].clone()
        qc_before = x[:,120:180].clone()
        qi_before = x[:,180:240].clone()
        qn_before = qc_before + qi_before
        
        x = self.preprocessing(x)
        x = self.original_model(x)
        x = self.postprocessing(x)
        
        t_new = t_before + x[:,0:60]*1200.
        qn_new = qn_before + x[:,120:180]*1200.
        liq_frac = self.apply_temperature_rules(t_new)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        xout = torch.zeros((x.shape[0],368), device = x.device)
        xout[:,0:120] = x[:,0:120]
        xout[:,240:] = x[:,180:]
        xout[:,120:180] = (qc_new - qc_before)/1200.
        xout[:,180:240] = (qi_new - qi_before)/1200.
        return xout

device = torch.device("cuda")
model_inf = modulus.Module.from_checkpoint(f_torch_model).to(device)
wrapped_model = WrappedModel(model_inf, input_sub, input_div, out_scale, lbd_qn).to(device)
WrappedModel.device = "cuda"
device = torch.device("cuda")
scripted_model = torch.jit.script(wrapped_model)
scripted_model = scripted_model.eval()
scripted_model.save(save_file_torch)

print('finished')