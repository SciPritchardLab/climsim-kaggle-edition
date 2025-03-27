import torch
import xarray as xr
import modulus
from climsim_utils.data_utils import *
from wrap_model import WrappedModel
import os

grid_info = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc')
input_mean = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_mean_v2_rh_mc_pervar.nc')
input_max = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_max_v2_rh_mc_pervar.nc')
input_min = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/input_min_v2_rh_mc_pervar.nc')
output_scale = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/output_scale_std_lowerthred_v2_rh_mc.nc')
qn_lbd = np.loadtxt('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/qn_exp_lambda_large.txt', delimiter=',')

data = data_utils(grid_info = grid_info, input_mean = input_mean, input_max = input_max, input_min = input_min, output_scale = output_scale)
data.set_to_v2_rh_mc_vars()
input_sub, input_div, out_scale = data.save_norm(write=False)

checkpoint_path = '/global/homes/j/jerrylin/scratch/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles/pure_resLSTM/pure_resLSTM_seed_43/ckpt/ckpt_epoch_9_metric_0.0727.mdlus'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# offline inference
save_path =  f'/global/homes/j/jerrylin/scratch/hugging/E3SM-MMF_ne4/saved_models/prelim_comparison/pure_resLSTM'
model = modulus.Module.from_checkpoint(checkpoint_path).to(device)
scripted_model = torch.jit.script(model)
scripted_model = scripted_model.eval()
scripted_model.save(os.path.join(save_path, 'model.pt'))

# online inference
wrapped_model = WrappedModel(original_model = model,
                            input_sub = torch.tensor(input_sub, dtype=torch.float32).to(device),
                            input_div = torch.tensor(input_div, dtype=torch.float32).to(device),
                            out_scale = torch.tensor(out_scale, dtype=torch.float32).to(device),
                            qn_lbd = torch.tensor(qn_lbd, dtype=torch.float32).to(device)).to(device)
scripted_model_wrapped = torch.jit.script(wrapped_model)
scripted_model_wrapped = scripted_model_wrapped.eval()
scripted_model_wrapped.save(os.path.join(save_path, 'wrapped_model.pt'))