from data_utils import *
import xarray as xr

grid_path = '../grid_info/ClimSim_low-res_grid-info.nc'
norm_path = '../preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v6_pervar.nc') # v2_rh_mc is a subset of v6, which itself is a subset of v5
input_max = xr.open_dataset(norm_path + 'inputs/input_max_v6_pervar.nc') # v2_rh_mc is a subset of v6, which itself is a subset of v5
input_min = xr.open_dataset(norm_path + 'inputs/input_min_v6_pervar.nc') # v2_rh_mc is a subset of v6, which itself is a subset of v5
output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_lowerthred_v6.nc') # v2_rh_mc is a subset of v6, which itself is a subset of v5

data = data_utils(grid_info = grid_info, 
                input_mean = input_mean, 
                input_max = input_max, 
                input_min = input_min, 
                output_scale = output_scale,
                qinput_log = False,
                normalize = False,
                input_abbrev='ml2steploc',
                output_abbrev='mlo') #!!!!!!! don't forget

data.data_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/train/'
data.set_to_v2_rh_mc_vars()

start_idx = 0
regexps = ['E3SM-MMF.ml2steploc.0001-02-14-*.nc']
data.set_regexps(data_split='scoring', regexps = regexps)
data.set_stride_sample(data_split='scoring', stride_sample=18)
data.set_filelist(data_split='scoring', start_idx=start_idx)
save_path = './'
data_loader = data.load_ncdata_with_generator('scoring')