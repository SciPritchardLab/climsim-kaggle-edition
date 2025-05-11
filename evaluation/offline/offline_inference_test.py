import numpy as np
from sklearn.metrics import r2_score
import torch
import os, gc
import modulus
from tqdm import tqdm
import sys
from climsim_utils.data_utils import *

grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

input_mean_v6_file = 'input_mean_v6_pervar.nc'
input_max_v6_file = 'input_max_v6_pervar.nc'
input_min_v6_file = 'input_min_v6_pervar.nc'
output_scale_v6_file = 'output_scale_std_lowerthred_v6.nc'

lbd_qn_file = 'qn_exp_lambda_large.txt'

grid_info = xr.open_dataset(grid_path)

input_mean_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_mean_v2_rh_mc_file)
input_max_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_max_v2_rh_mc_file)
input_min_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_min_v2_rh_mc_file)
output_scale_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_v2_rh_mc_file)

input_mean_v6 = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_mean_v6_file)
input_max_v6 = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_max_v6_file)
input_min_v6 = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_min_v6_file)
output_scale_v6 = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_v6_file)

lbd_qn = np.loadtxt('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + lbd_qn_file, delimiter = ',')

data_v2_rh_mc = data_utils(grid_info = grid_info, 
                           input_mean = input_mean_v2_rh_mc, 
                           input_max = input_max_v2_rh_mc, 
                           input_min = input_min_v2_rh_mc, 
                           output_scale = output_scale_v2_rh_mc,
                           qinput_log = False,
                           normalize = False)
data_v2_rh_mc.set_to_v2_rh_mc_vars()

data_v6 = data_utils(grid_info = grid_info,
                     input_mean = input_mean_v6,
                     input_max = input_max_v6,
                     input_min = input_min_v6,
                     output_scale = output_scale_v6,
                     qinput_log = False,
                     normalize = False)                     
data_v6.set_to_v6_vars()

input_sub_v2_rh_mc, input_div_v2_rh_mc, out_scale_v2_rh_mc = data_v2_rh_mc.save_norm(write=False) # this extracts only the relevant variables
input_sub_v2_rh_mc = input_sub_v2_rh_mc[None, :]
input_div_v2_rh_mc = input_div_v2_rh_mc[None, :]
out_scale_v2_rh_mc = out_scale_v2_rh_mc[None, :]

input_sub_v6, input_div_v6, out_scale_v6 = data_v6.save_norm(write=False) # this extracts only the relevant variables
input_sub_v6 = input_sub_v6[None, :]
input_div_v6 = input_div_v6[None, :]
out_scale_v6 = out_scale_v6[None, :]

per_lev_sub = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_sub.nc')
per_lev_div = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_div.nc')
per_col_sub = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_col_sub.nc')
per_col_div = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_col_div.nc')
per_lev_min_norm = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/multirepresentation/per_lev_min_norm.nc')
input_sub_per_lev, input_div_per_lev, input_sub_per_col, input_div_per_col, input_min_norm_per_lev, out_scale = data_v2_rh_mc.save_norm_multirep(per_lev_sub,
                                                                                                                                                 per_lev_div,
                                                                                                                                                 per_col_sub,
                                                                                                                                                 per_col_div,
                                                                                                                                                 per_lev_min_norm,
                                                                                                                                                 save_path = '',                    
                                                                                                                                                 write=False)

lat = grid_info['lat'].values
lon = grid_info['lon'].values
lat_bin_mids = data_v2_rh_mc.lat_bin_mids

v2_rh_mc_input_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_input.npy'
v2_rh_mc_target_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_target.npy'
standard_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/standard/'
conf_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/conf_loss/'
diff_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/diff_loss/'
multirep_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/multirep/'

v6_input_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/test_input.npy'
v6_target_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/test_target.npy'
v6_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/test_preds/'

def apply_temperature_rules(T):
    # Create an output tensor, initialized to zero
    output = np.zeros_like(T)

    # Apply the linear transition within the range 253.16 to 273.16
    mask = (T >= 253.16) & (T <= 273.16)
    output[mask] = (T[mask] - 253.16) / 20.0  # 20.0 is the range (273.16 - 253.16)

    # Values where T > 273.16 set to 1
    output[T > 273.16] = 1

    # Values where T < 253.16 are already set to 0 by the initialization
    return output

def preprocessing_v2_rh_mc(data, input_path, target_path, input_sub, input_div, lbd_qn, out_scale):
    npy_input = np.load(input_path)
    npy_target = np.load(target_path)

    surface_pressure = npy_input[:, data.ps_index]
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    actual_input = npy_input.copy().reshape(-1, data.num_latlon, data.input_feature_len)

    npy_input[:,120:180] = 1 - np.exp(-npy_input[:,120:180] * lbd_qn)
    npy_input = (npy_input - input_sub)/input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:,120:120+15] = 0
    npy_input[:,60:120] = np.clip(npy_input[:,60:120], 0, 1.2)
    torch_input = torch.tensor(npy_input).float()

    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + reshaped_target[:, :, 0:60]*1200
    qn_new = qn_before + reshaped_target[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_target = np.concatenate((reshaped_target[:, :, 0:120], 
                                    (qc_new - qc_before)/1200, 
                                    (qi_new - qi_before)/1200, 
                                    reshaped_target[:, :, 180:240], 
                                    reshaped_target[:, :, 240:]), axis=2)
    return torch_input, actual_input, actual_target, pressures_binned

def preprocessing_v6(data, input_path, target_path, input_sub, input_div, lbd_qn, out_scale):
    npy_input = np.load(input_path)
    npy_target = np.load(target_path)
    
    surface_pressure = npy_input[:, data.ps_index]
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    actual_input = npy_input.copy().reshape(-1, data.num_latlon, data.input_feature_len)

    npy_input[:,120:180] = 1 - np.exp(-npy_input[:,120:180] * lbd_qn)
    npy_input = (npy_input - input_sub)/input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:,120:120+15] = 0
    npy_input[:,60:120] = np.clip(npy_input[:,60:120], 0, 1.2)
    torch_input = torch.tensor(npy_input).float()

    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + reshaped_target[:, :, 0:60]*1200
    qn_new = qn_before + reshaped_target[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_target = np.concatenate((reshaped_target[:, :, 0:120], 
                                    (qc_new - qc_before)/1200, 
                                    (qi_new - qi_before)/1200, 
                                    reshaped_target[:, :, 180:240], 
                                    reshaped_target[:, :, 240:]), axis=2)
    return torch_input, actual_input, actual_target, pressures_binned

def preprocessing_multirep(data,
                           input_path,
                           target_path,
                           input_sub_per_lev,
                           input_div_per_lev,
                           input_sub_per_col,
                           input_div_per_col,
                           input_min_norm_per_lev,
                           out_scale):
    npy_input = np.load(input_path)
    npy_target = np.load(target_path)
    
    surface_pressure = npy_input[:, data.ps_index]
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    actual_input = npy_input.copy().reshape(-1, data.num_latlon, data.input_feature_len)

    x1 = (npy_input - input_sub_per_lev)/input_div_per_lev
    x1 = np.where(np.isnan(x1), 0, x1)
    x1 = np.where(np.isinf(x1), 0, x1)
    x2 = (np.concatenate([npy_input[:,:180], npy_input[:,240:540]], axis = 1) - input_sub_per_col)/input_div_per_col
    x2 = np.where(np.isnan(x2), 0, x2)
    x2 = np.where(np.isinf(x2), 0, x2)
    x_col_norm = np.concatenate([npy_input[:,:180], npy_input[:,240:540]], axis=1)
    x3 = np.where(x_col_norm >= input_min_norm_per_lev, \
                  np.log((x_col_norm - input_min_norm_per_lev) + 1), \
                  -np.log((input_min_norm_per_lev - x_col_norm) + 1))
    npy_input = np.concatenate([x1[:,:540], x2, x3, x1[:,540:]], axis=1)

    torch_input = torch.tensor(npy_input).float()

    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + reshaped_target[:, :, 0:60]*1200
    qn_new = qn_before + reshaped_target[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_target = np.concatenate((reshaped_target[:, :, 0:120], 
                                    (qc_new - qc_before)/1200, 
                                    (qi_new - qi_before)/1200, 
                                    reshaped_target[:, :, 180:240], 
                                    reshaped_target[:, :, 240:]), axis=2)
    return torch_input, actual_input, actual_target, pressures_binned

# standard models
standard_unet_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet/unet_seed_7/model.pt'
standard_unet_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet/unet_seed_43/model.pt'
standard_unet_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet/unet_seed_1024/model.pt'
standard_squeezeformer_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer/squeezeformer_seed_7/model.pt'
standard_squeezeformer_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer/squeezeformer_seed_43/model.pt'
standard_squeezeformer_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer/squeezeformer_seed_1024/model.pt'
standard_pure_resLSTM_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM/pure_resLSTM_seed_7/model.pt'
standard_pure_resLSTM_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM/pure_resLSTM_seed_43/model.pt'
standard_pure_resLSTM_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM/pure_resLSTM_seed_1024/model.pt'
standard_pao_model_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model/pao_model_seed_7/model.pt'
standard_pao_model_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model/pao_model_seed_43/model.pt'
standard_pao_model_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model/pao_model_seed_1024/model.pt'
standard_convnext_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext/convnext_seed_7/model.pt'
standard_convnext_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext/convnext_seed_43/model.pt'
standard_convnext_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext/convnext_seed_1024/model.pt'
standard_encdec_lstm_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm/encdec_lstm_seed_7/model.pt'
standard_encdec_lstm_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm/encdec_lstm_seed_43/model.pt'
standard_encdec_lstm_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm/encdec_lstm_seed_1024/model.pt'

# confidence loss models
conf_loss_unet_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_conf/unet_conf_seed_7/model.pt'
conf_loss_unet_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_conf/unet_conf_seed_43/model.pt'
conf_loss_unet_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_conf/unet_conf_seed_1024/model.pt'
conf_loss_squeezeformer_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_conf/squeezeformer_conf_seed_7/model.pt'
conf_loss_squeezeformer_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_conf/squeezeformer_conf_seed_43/model.pt'
conf_loss_squeezeformer_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_conf/squeezeformer_conf_seed_1024/model.pt'
conf_loss_pure_resLSTM_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_conf/pure_resLSTM_conf_seed_7/model.pt'
conf_loss_pure_resLSTM_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_conf/pure_resLSTM_conf_seed_43/model.pt'
conf_loss_pure_resLSTM_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_conf/pure_resLSTM_conf_seed_1024/model.pt'
conf_loss_pao_model_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_conf/pao_model_conf_seed_7/model.pt'
conf_loss_pao_model_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_conf/pao_model_conf_seed_43/model.pt'
conf_loss_pao_model_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_conf/pao_model_conf_seed_1024/model.pt'
conf_loss_convnext_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_conf/convnext_conf_seed_7/model.pt'
conf_loss_convnext_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_conf/convnext_conf_seed_43/model.pt'
conf_loss_convnext_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_conf/convnext_conf_seed_1024/model.pt'
conf_loss_encdec_lstm_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_conf/encdec_lstm_conf_seed_7/model.pt'
conf_loss_encdec_lstm_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_conf/encdec_lstm_conf_seed_43/model.pt'
conf_loss_encdec_lstm_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_conf/encdec_lstm_conf_seed_1024/model.pt'

# diff loss models
diff_loss_unet_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_diff/unet_diff_seed_7/model.pt'
diff_loss_unet_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_diff/unet_diff_seed_43/model.pt'
diff_loss_unet_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_diff/unet_diff_seed_1024/model.pt'
diff_loss_squeezeformer_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_diff/squeezeformer_diff_seed_7/model.pt'
diff_loss_squeezeformer_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_diff/squeezeformer_diff_seed_43/model.pt'
diff_loss_squeezeformer_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_diff/squeezeformer_diff_seed_1024/model.pt'
diff_loss_pure_resLSTM_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_diff/pure_resLSTM_diff_seed_7/model.pt'
diff_loss_pure_resLSTM_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_diff/pure_resLSTM_diff_seed_43/model.pt'
diff_loss_pure_resLSTM_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_diff/pure_resLSTM_diff_seed_1024/model.pt'
diff_loss_pao_model_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_diff/pao_model_diff_seed_7/model.pt'
diff_loss_pao_model_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_diff/pao_model_diff_seed_43/model.pt'
diff_loss_pao_model_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_diff/pao_model_diff_seed_1024/model.pt'
diff_loss_convnext_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_diff/convnext_diff_seed_7/model.pt'
diff_loss_convnext_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_diff/convnext_diff_seed_43/model.pt'
diff_loss_convnext_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_diff/convnext_diff_seed_1024/model.pt'
diff_loss_encdec_lstm_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_diff/encdec_lstm_diff_seed_7/model.pt'
diff_loss_encdec_lstm_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_diff/encdec_lstm_diff_seed_43/model.pt'
diff_loss_encdec_lstm_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_diff/encdec_lstm_diff_seed_1024/model.pt'

# multirep models
multirep_unet_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_multirep/unet_multirep_seed_7/model.pt'
multirep_unet_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_multirep/unet_multirep_seed_43/model.pt'
multirep_unet_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_multirep/unet_multirep_seed_1024/model.pt'
multirep_squeezeformer_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_multirep/squeezeformer_multirep_seed_7/model.pt'
multirep_squeezeformer_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_multirep/squeezeformer_multirep_seed_43/model.pt'
multirep_squeezeformer_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_multirep/squeezeformer_multirep_seed_1024/model.pt'
multirep_pure_resLSTM_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_multirep/pure_resLSTM_multirep_seed_7/model.pt'
multirep_pure_resLSTM_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_multirep/pure_resLSTM_multirep_seed_43/model.pt'
multirep_pure_resLSTM_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_multirep/pure_resLSTM_multirep_seed_1024/model.pt'
multirep_pao_model_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_multirep/pao_model_multirep_seed_7/model.pt'
multirep_pao_model_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_multirep/pao_model_multirep_seed_43/model.pt'
multirep_pao_model_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_multirep/pao_model_multirep_seed_1024/model.pt'
multirep_convnext_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_multirep/convnext_multirep_seed_7/model.pt'
multirep_convnext_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_multirep/convnext_multirep_seed_43/model.pt'
multirep_convnext_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_multirep/convnext_multirep_seed_1024/model.pt'
multirep_encdec_lstm_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_multirep/encdec_lstm_multirep_seed_7/model.pt'
multirep_encdec_lstm_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_multirep/encdec_lstm_multirep_seed_43/model.pt'
multirep_encdec_lstm_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_multirep/encdec_lstm_multirep_seed_1024/model.pt'

# v6 models
v6_unet_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_v6/unet_v6_seed_7/model.pt'
v6_unet_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_v6/unet_v6_seed_43/model.pt'
v6_unet_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/unet_v6/unet_v6_seed_1024/model.pt'
v6_squeezeformer_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_v6/squeezeformer_v6_seed_7/model.pt'
v6_squeezeformer_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_v6/squeezeformer_v6_seed_43/model.pt'
v6_squeezeformer_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/squeezeformer_v6/squeezeformer_v6_seed_1024/model.pt'
v6_pure_resLSTM_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_v6/pure_resLSTM_v6_seed_7/model.pt'
v6_pure_resLSTM_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_v6/pure_resLSTM_v6_seed_43/model.pt'
v6_pure_resLSTM_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pure_resLSTM_v6/pure_resLSTM_v6_seed_1024/model.pt'
v6_pao_model_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_v6/pao_model_v6_seed_7/model.pt'
v6_pao_model_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_v6/pao_model_v6_seed_43/model.pt'
v6_pao_model_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/pao_model_v6/pao_model_v6_seed_1024/model.pt'
v6_convnext_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_v6/convnext_v6_seed_7/model.pt'
v6_convnext_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_v6/convnext_v6_seed_43/model.pt'
v6_convnext_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/convnext_v6/convnext_v6_seed_1024/model.pt'
v6_encdec_lstm_seed_7_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_v6/encdec_lstm_v6_seed_7/model.pt'
v6_encdec_lstm_seed_43_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_v6/encdec_lstm_v6_seed_43/model.pt'
v6_encdec_lstm_seed_1024_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/encdec_lstm_v6/encdec_lstm_v6_seed_1024/model.pt'

standard_model_paths = {
    'standard_unet_seed_7': standard_unet_seed_7_path,
    'standard_unet_seed_43': standard_unet_seed_43_path,
    'standard_unet_seed_1024': standard_unet_seed_1024_path,
    'standard_squeezeformer_seed_7': standard_squeezeformer_seed_7_path,
    'standard_squeezeformer_seed_43': standard_squeezeformer_seed_43_path,
    'standard_squeezeformer_seed_1024': standard_squeezeformer_seed_1024_path,
    'standard_pure_resLSTM_seed_7': standard_pure_resLSTM_seed_7_path,
    'standard_pure_resLSTM_seed_43': standard_pure_resLSTM_seed_43_path,
    'standard_pure_resLSTM_seed_1024': standard_pure_resLSTM_seed_1024_path,
    'standard_pao_model_seed_7': standard_pao_model_seed_7_path,
    'standard_pao_model_seed_43': standard_pao_model_seed_43_path,
    'standard_pao_model_seed_1024': standard_pao_model_seed_1024_path,
    'standard_convnext_seed_7': standard_convnext_seed_7_path,
    'standard_convnext_seed_43': standard_convnext_seed_43_path,
    'standard_convnext_seed_1024': standard_convnext_seed_1024_path,
    'standard_encdec_lstm_seed_7': standard_encdec_lstm_seed_7_path,
    'standard_encdec_lstm_seed_43': standard_encdec_lstm_seed_43_path,
    'standard_encdec_lstm_seed_1024': standard_encdec_lstm_seed_1024_path,
}

conf_loss_model_paths = {
    'conf_loss_unet_seed_7': conf_loss_unet_seed_7_path,
    'conf_loss_unet_seed_43': conf_loss_unet_seed_43_path,
    'conf_loss_unet_seed_1024': conf_loss_unet_seed_1024_path,
    'conf_loss_squeezeformer_seed_7': conf_loss_squeezeformer_seed_7_path,
    'conf_loss_squeezeformer_seed_43': conf_loss_squeezeformer_seed_43_path,
    'conf_loss_squeezeformer_seed_1024': conf_loss_squeezeformer_seed_1024_path,
    'conf_loss_pure_resLSTM_seed_7': conf_loss_pure_resLSTM_seed_7_path,
    'conf_loss_pure_resLSTM_seed_43': conf_loss_pure_resLSTM_seed_43_path,
    'conf_loss_pure_resLSTM_seed_1024': conf_loss_pure_resLSTM_seed_1024_path,
    'conf_loss_pao_model_seed_7': conf_loss_pao_model_seed_7_path,
    'conf_loss_pao_model_seed_43': conf_loss_pao_model_seed_43_path,
    'conf_loss_pao_model_seed_1024': conf_loss_pao_model_seed_1024_path,
    'conf_loss_convnext_seed_7': conf_loss_convnext_seed_7_path,
    'conf_loss_convnext_seed_43': conf_loss_convnext_seed_43_path,
    'conf_loss_convnext_seed_1024': conf_loss_convnext_seed_1024_path,
    'conf_loss_encdec_lstm_seed_7': conf_loss_encdec_lstm_seed_7_path,
    'conf_loss_encdec_lstm_seed_43': conf_loss_encdec_lstm_seed_43_path,
    'conf_loss_encdec_lstm_seed_1024': conf_loss_encdec_lstm_seed_1024_path
}

diff_loss_model_paths = {
    'diff_loss_unet_seed_7': diff_loss_unet_seed_7_path,
    'diff_loss_unet_seed_43': diff_loss_unet_seed_43_path,
    'diff_loss_unet_seed_1024': diff_loss_unet_seed_1024_path,
    'diff_loss_squeezeformer_seed_7': diff_loss_squeezeformer_seed_7_path,
    'diff_loss_squeezeformer_seed_43': diff_loss_squeezeformer_seed_43_path,
    'diff_loss_squeezeformer_seed_1024': diff_loss_squeezeformer_seed_1024_path,
    'diff_loss_pure_resLSTM_seed_7': diff_loss_pure_resLSTM_seed_7_path,
    'diff_loss_pure_resLSTM_seed_43': diff_loss_pure_resLSTM_seed_43_path,
    'diff_loss_pure_resLSTM_seed_1024': diff_loss_pure_resLSTM_seed_1024_path,
    'diff_loss_pao_model_seed_7': diff_loss_pao_model_seed_7_path,
    'diff_loss_pao_model_seed_43': diff_loss_pao_model_seed_43_path,
    'diff_loss_pao_model_seed_1024': diff_loss_pao_model_seed_1024_path,
    'diff_loss_convnext_seed_7': diff_loss_convnext_seed_7_path,
    'diff_loss_convnext_seed_43': diff_loss_convnext_seed_43_path,
    'diff_loss_convnext_seed_1024': diff_loss_convnext_seed_1024_path,
    'diff_loss_encdec_lstm_seed_7': diff_loss_encdec_lstm_seed_7_path,
    'diff_loss_encdec_lstm_seed_43': diff_loss_encdec_lstm_seed_43_path,
    'diff_loss_encdec_lstm_seed_1024': diff_loss_encdec_lstm_seed_1024_path
}

multirep_model_paths = {
    'multirep_unet_seed_7': multirep_unet_seed_7_path,
    'multirep_unet_seed_43': multirep_unet_seed_43_path,
    'multirep_unet_seed_1024': multirep_unet_seed_1024_path,
    'multirep_squeezeformer_seed_7': multirep_squeezeformer_seed_7_path,
    'multirep_squeezeformer_seed_43': multirep_squeezeformer_seed_43_path,
    'multirep_squeezeformer_seed_1024': multirep_squeezeformer_seed_1024_path,
    'multirep_pure_resLSTM_seed_7': multirep_pure_resLSTM_seed_7_path,
    'multirep_pure_resLSTM_seed_43': multirep_pure_resLSTM_seed_43_path,
    'multirep_pure_resLSTM_seed_1024': multirep_pure_resLSTM_seed_1024_path,
    'multirep_pao_model_seed_7': multirep_pao_model_seed_7_path,
    'multirep_pao_model_seed_43': multirep_pao_model_seed_43_path,
    'multirep_pao_model_seed_1024': multirep_pao_model_seed_1024_path,
    'multirep_convnext_seed_7': multirep_convnext_seed_7_path,
    'multirep_convnext_seed_43': multirep_convnext_seed_43_path,
    'multirep_convnext_seed_1024': multirep_convnext_seed_1024_path,
    'multirep_encdec_lstm_seed_7': multirep_encdec_lstm_seed_7_path,
    'multirep_encdec_lstm_seed_43': multirep_encdec_lstm_seed_43_path,
    'multirep_encdec_lstm_seed_1024': multirep_encdec_lstm_seed_1024_path
}

v6_model_paths = {
    'v6_unet_seed_7': v6_unet_seed_7_path,
    'v6_unet_seed_43': v6_unet_seed_43_path,
    'v6_unet_seed_1024': v6_unet_seed_1024_path,
    'v6_squeezeformer_seed_7': v6_squeezeformer_seed_7_path,
    'v6_squeezeformer_seed_43': v6_squeezeformer_seed_43_path,
    'v6_squeezeformer_seed_1024': v6_squeezeformer_seed_1024_path,
    'v6_pure_resLSTM_seed_7': v6_pure_resLSTM_seed_7_path,
    'v6_pure_resLSTM_seed_43': v6_pure_resLSTM_seed_43_path,
    'v6_pure_resLSTM_seed_1024': v6_pure_resLSTM_seed_1024_path,
    'v6_pao_model_seed_7': v6_pao_model_seed_7_path,
    'v6_pao_model_seed_43': v6_pao_model_seed_43_path,
    'v6_pao_model_seed_1024': v6_pao_model_seed_1024_path,
    'v6_convnext_seed_7': v6_convnext_seed_7_path,
    'v6_convnext_seed_43': v6_convnext_seed_43_path,
    'v6_convnext_seed_1024': v6_convnext_seed_1024_path,
    'v6_encdec_lstm_seed_7': v6_encdec_lstm_seed_7_path,
    'v6_encdec_lstm_seed_43': v6_encdec_lstm_seed_43_path,
    'v6_encdec_lstm_seed_1024': v6_encdec_lstm_seed_1024_path
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_model(data, model_path, actual_input, torch_input, out_scale):
    model = torch.jit.load(model_path).to(device)
    model.eval()
    batch_pred_list = []
    batch_size = data.num_latlon
    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size)):
            batch = torch_input[i:i+batch_size].to(device)
            batch_pred = model(batch)
            batch_pred[:, 60:75] = 0
            batch_pred[:, 120:135] = 0
            batch_pred[:, 180:195] = 0
            batch_pred[:, 240:255] = 0
            batch_pred_list.append(batch_pred.cpu().numpy() / out_scale)
    model_preds = np.stack(batch_pred_list, axis=0)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + model_preds[:, :, 0:60]*1200
    qn_new = qn_before + model_preds[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_preds = np.concatenate((model_preds[:, :, 0:120], 
                                  (qc_new - qc_before)/1200, 
                                  (qi_new - qi_before)/1200, 
                                   model_preds[:, :, 180:240], 
                                   model_preds[:, :, 240:]), axis=2)

    del model
    del batch_pred_list
    gc.collect()
    torch.cuda.empty_cache()
    return actual_preds

def inference_model_conf_loss(data, model_path, actual_input, torch_input, out_scale):
    model = torch.jit.load(model_path).to(device)
    model.eval()
    batch_pred_list = []
    batch_conf_list = []
    batch_size = data.num_latlon
    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size)):
            batch = torch_input[i:i+batch_size].to(device)
            batch_pred, batch_conf = model(batch)
            batch_pred[:, 60:75] = 0
            batch_pred[:, 120:135] = 0
            batch_pred[:, 180:195] = 0
            batch_pred[:, 240:255] = 0
            batch_pred_list.append(batch_pred.cpu().numpy() / out_scale)
            batch_conf_list.append(batch_conf.cpu().numpy())
    model_preds = np.stack(batch_pred_list, axis=0)
    model_conf = np.stack(batch_conf_list, axis=0)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + model_preds[:, :, 0:60]*1200
    qn_new = qn_before + model_preds[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_preds = np.concatenate((model_preds[:, :, 0:120], 
                                  (qc_new - qc_before)/1200, 
                                  (qi_new - qi_before)/1200, 
                                   model_preds[:, :, 180:240], 
                                   model_preds[:, :, 240:]), axis=2)

    del model
    del batch_pred_list
    del batch_conf_list
    gc.collect()
    torch.cuda.empty_cache()
    return actual_preds, model_conf

torch_input_v2_rh_mc, actual_input_v2_rh_mc, actual_target, pressures_binned = preprocessing_v2_rh_mc(data = data_v2_rh_mc, 
                                                                                                      input_path = v2_rh_mc_input_path, 
                                                                                                      target_path = v2_rh_mc_target_path, 
                                                                                                      input_sub = input_sub_v2_rh_mc, 
                                                                                                      input_div = input_div_v2_rh_mc, 
                                                                                                      lbd_qn = lbd_qn, 
                                                                                                      out_scale = out_scale_v2_rh_mc)

torch_input_v6, actual_input_v6, actual_target, pressures_binned = preprocessing_v6(data = data_v6, 
                                                                                    input_path = v6_input_path, 
                                                                                    target_path = v6_target_path, 
                                                                                    input_sub = input_sub_v6, 
                                                                                    input_div = input_div_v6, 
                                                                                    lbd_qn = lbd_qn, 
                                                                                    out_scale = out_scale_v6)


torch_input_multirep, actual_input_multirep, actual_target, pressures_binned = preprocessing_multirep(data = data_v2_rh_mc,
                                                                                                      input_path = v2_rh_mc_input_path,
                                                                                                      target_path = v2_rh_mc_target_path,
                                                                                                      input_sub_per_lev = input_sub_per_lev,
                                                                                                      input_div_per_lev = input_div_per_lev,
                                                                                                      input_sub_per_col = input_sub_per_col,
                                                                                                      input_div_per_col = input_div_per_col,
                                                                                                      input_min_norm_per_lev = input_min_norm_per_lev,
                                                                                                      out_scale = out_scale)


np.save('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/actual_input.npy', actual_input_v2_rh_mc)
np.save('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/actual_target.npy', actual_target)

np.save('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/actual_input.npy', actual_input_v6)
np.save('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/actual_target.npy', actual_target)

# standard u-net
print("Running standard u-net inference...")
standard_unet_preds_1 = inference_model(data_v2_rh_mc,
                                        standard_model_paths['standard_unet_seed_7'],
                                        actual_input_v2_rh_mc,
                                        torch_input_v2_rh_mc,
                                        out_scale_v2_rh_mc)

standard_unet_preds_2 = inference_model(data_v2_rh_mc,
                                        standard_model_paths['standard_unet_seed_43'],
                                        actual_input_v2_rh_mc,
                                        torch_input_v2_rh_mc,
                                        out_scale_v2_rh_mc)

standard_unet_preds_3 = inference_model(data_v2_rh_mc,
                                        standard_model_paths['standard_unet_seed_1024'],
                                        actual_input_v2_rh_mc,
                                        torch_input_v2_rh_mc,
                                        out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_unet_preds.npz'), 
         seed_7 = standard_unet_preds_1, 
         seed_43 = standard_unet_preds_2, 
         seed_1024 = standard_unet_preds_3)

del standard_unet_preds_1
del standard_unet_preds_2
del standard_unet_preds_3
gc.collect()

# standard squeezeformer
print("Running standard squeezeformer inference...")
standard_squeezeformer_preds_1 = inference_model(data_v2_rh_mc,
                                                 standard_model_paths['standard_squeezeformer_seed_7'],
                                                 actual_input_v2_rh_mc,
                                                 torch_input_v2_rh_mc,
                                                 out_scale_v2_rh_mc)

standard_squeezeformer_preds_2 = inference_model(data_v2_rh_mc,
                                                 standard_model_paths['standard_squeezeformer_seed_43'],
                                                 actual_input_v2_rh_mc,
                                                 torch_input_v2_rh_mc,
                                                 out_scale_v2_rh_mc)

standard_squeezeformer_preds_3 = inference_model(data_v2_rh_mc,
                                                 standard_model_paths['standard_squeezeformer_seed_1024'],
                                                 actual_input_v2_rh_mc,
                                                 torch_input_v2_rh_mc,
                                                 out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_squeezeformer_preds.npz'),
         seed_7 = standard_squeezeformer_preds_1, 
         seed_43 = standard_squeezeformer_preds_2, 
         seed_1024 = standard_squeezeformer_preds_3)

del standard_squeezeformer_preds_1
del standard_squeezeformer_preds_2
del standard_squeezeformer_preds_3
gc.collect()

# standard pure_resLSTM
print("Running standard pure_resLSTM inference...")
standard_pure_resLSTM_preds_1 = inference_model(data_v2_rh_mc,
                                                 standard_model_paths['standard_pure_resLSTM_seed_7'],
                                                 actual_input_v2_rh_mc,
                                                 torch_input_v2_rh_mc,
                                                 out_scale_v2_rh_mc)

standard_pure_resLSTM_preds_2 = inference_model(data_v2_rh_mc,
                                                standard_model_paths['standard_pure_resLSTM_seed_43'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

standard_pure_resLSTM_preds_3 = inference_model(data_v2_rh_mc,
                                                standard_model_paths['standard_pure_resLSTM_seed_1024'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_pure_resLSTM_preds.npz'),
         seed_7 = standard_pure_resLSTM_preds_1, 
         seed_43 = standard_pure_resLSTM_preds_2, 
         seed_1024 = standard_pure_resLSTM_preds_3)

del standard_pure_resLSTM_preds_1
del standard_pure_resLSTM_preds_2
del standard_pure_resLSTM_preds_3
gc.collect()

# standard pao_model
print("Running standard pao_model inference...")
standard_pao_model_preds_1 = inference_model(data_v2_rh_mc,
                                             standard_model_paths['standard_pao_model_seed_7'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

standard_pao_model_preds_2 = inference_model(data_v2_rh_mc,
                                             standard_model_paths['standard_pao_model_seed_43'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

standard_pao_model_preds_3 = inference_model(data_v2_rh_mc,
                                             standard_model_paths['standard_pao_model_seed_1024'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_pao_model_preds.npz'),
         seed_7 = standard_pao_model_preds_1, 
         seed_43 = standard_pao_model_preds_2, 
         seed_1024 = standard_pao_model_preds_3)

del standard_pao_model_preds_1
del standard_pao_model_preds_2
del standard_pao_model_preds_3
gc.collect()

# standard convnext
print("Running standard convnext inference...")
standard_convnext_preds_1 = inference_model(data_v2_rh_mc,
                                            standard_model_paths['standard_convnext_seed_7'],
                                            actual_input_v2_rh_mc,
                                            torch_input_v2_rh_mc,
                                            out_scale_v2_rh_mc)

standard_convnext_preds_2 = inference_model(data_v2_rh_mc,
                                            standard_model_paths['standard_convnext_seed_43'],
                                            actual_input_v2_rh_mc,
                                            torch_input_v2_rh_mc,
                                            out_scale_v2_rh_mc)

standard_convnext_preds_3 = inference_model(data_v2_rh_mc,
                                            standard_model_paths['standard_convnext_seed_1024'],
                                            actual_input_v2_rh_mc,
                                            torch_input_v2_rh_mc,
                                            out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_convnext_preds.npz'),
         seed_7 = standard_convnext_preds_1, 
         seed_43 = standard_convnext_preds_2, 
         seed_1024 = standard_convnext_preds_3)

del standard_convnext_preds_1
del standard_convnext_preds_2
del standard_convnext_preds_3
gc.collect()

# standard encdec_lstm
print("Running standard encdec_lstm inference...")
standard_encdec_lstm_preds_1 = inference_model(data_v2_rh_mc,
                                                standard_model_paths['standard_encdec_lstm_seed_7'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

standard_encdec_lstm_preds_2 = inference_model(data_v2_rh_mc,
                                                standard_model_paths['standard_encdec_lstm_seed_43'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

standard_encdec_lstm_preds_3 = inference_model(data_v2_rh_mc,
                                                standard_model_paths['standard_encdec_lstm_seed_1024'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(standard_save_path, 'standard_encdec_lstm_preds.npz'),
         seed_7 = standard_encdec_lstm_preds_1, 
         seed_43 = standard_encdec_lstm_preds_2, 
         seed_1024 = standard_encdec_lstm_preds_3)

del standard_encdec_lstm_preds_1
del standard_encdec_lstm_preds_2
del standard_encdec_lstm_preds_3
gc.collect()

# conf loss unet
print("Running conf loss unet inference...")
conf_loss_unet_preds_1, conf_loss_unet_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                          conf_loss_model_paths['conf_loss_unet_seed_7'],
                                                                          actual_input_v2_rh_mc,
                                                                          torch_input_v2_rh_mc,
                                                                          out_scale_v2_rh_mc)

conf_loss_unet_preds_2, conf_loss_unet_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                          conf_loss_model_paths['conf_loss_unet_seed_43'],
                                                                          actual_input_v2_rh_mc,
                                                                          torch_input_v2_rh_mc,
                                                                          out_scale_v2_rh_mc)

conf_loss_unet_preds_3, conf_loss_unet_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                          conf_loss_model_paths['conf_loss_unet_seed_1024'],
                                                                          actual_input_v2_rh_mc,
                                                                          torch_input_v2_rh_mc,
                                                                          out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_unet_preds.npz'),
         seed_7 = conf_loss_unet_preds_1, 
         seed_43 = conf_loss_unet_preds_2, 
         seed_1024 = conf_loss_unet_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_unet_conf.npz'),
         seed_7 = conf_loss_unet_conf_1, 
         seed_43 = conf_loss_unet_conf_2, 
         seed_1024 = conf_loss_unet_conf_3)

del conf_loss_unet_preds_1
del conf_loss_unet_preds_2
del conf_loss_unet_preds_3
del conf_loss_unet_conf_1
del conf_loss_unet_conf_2
del conf_loss_unet_conf_3
gc.collect()

# conf loss squeezeformer
print("Running conf loss squeezeformer inference...")
conf_loss_squeezeformer_preds_1, conf_loss_squeezeformer_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                            conf_loss_model_paths['conf_loss_squeezeformer_seed_7'],
                                                                                            actual_input_v2_rh_mc,
                                                                                            torch_input_v2_rh_mc,
                                                                                            out_scale_v2_rh_mc)

conf_loss_squeezeformer_preds_2, conf_loss_squeezeformer_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                            conf_loss_model_paths['conf_loss_squeezeformer_seed_43'],
                                                                                            actual_input_v2_rh_mc,
                                                                                            torch_input_v2_rh_mc,
                                                                                            out_scale_v2_rh_mc)

conf_loss_squeezeformer_preds_3, conf_loss_squeezeformer_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                            conf_loss_model_paths['conf_loss_squeezeformer_seed_1024'],
                                                                                            actual_input_v2_rh_mc,
                                                                                            torch_input_v2_rh_mc,
                                                                                            out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_squeezeformer_preds.npz'),
         seed_7 = conf_loss_squeezeformer_preds_1, 
         seed_43 = conf_loss_squeezeformer_preds_2, 
         seed_1024 = conf_loss_squeezeformer_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_squeezeformer_conf.npz'),
         seed_7 = conf_loss_squeezeformer_conf_1, 
         seed_43 = conf_loss_squeezeformer_conf_2, 
         seed_1024 = conf_loss_squeezeformer_conf_3)

del conf_loss_squeezeformer_preds_1
del conf_loss_squeezeformer_preds_2
del conf_loss_squeezeformer_preds_3
del conf_loss_squeezeformer_conf_1
del conf_loss_squeezeformer_conf_2
del conf_loss_squeezeformer_conf_3
gc.collect()

# conf loss pure_resLSTM
print("Running conf loss pure_resLSTM inference...")
conf_loss_pure_resLSTM_preds_1, conf_loss_pure_resLSTM_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                          conf_loss_model_paths['conf_loss_pure_resLSTM_seed_7'],
                                                                                          actual_input_v2_rh_mc,
                                                                                          torch_input_v2_rh_mc,
                                                                                          out_scale_v2_rh_mc)

conf_loss_pure_resLSTM_preds_2, conf_loss_pure_resLSTM_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                          conf_loss_model_paths['conf_loss_pure_resLSTM_seed_43'],
                                                                                          actual_input_v2_rh_mc,
                                                                                          torch_input_v2_rh_mc,
                                                                                          out_scale_v2_rh_mc)

conf_loss_pure_resLSTM_preds_3, conf_loss_pure_resLSTM_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                          conf_loss_model_paths['conf_loss_pure_resLSTM_seed_1024'],
                                                                                          actual_input_v2_rh_mc,
                                                                                          torch_input_v2_rh_mc,
                                                                                          out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_pure_resLSTM_preds.npz'),
            seed_7 = conf_loss_pure_resLSTM_preds_1, 
            seed_43 = conf_loss_pure_resLSTM_preds_2, 
            seed_1024 = conf_loss_pure_resLSTM_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_pure_resLSTM_conf.npz'),
            seed_7 = conf_loss_pure_resLSTM_conf_1, 
            seed_43 = conf_loss_pure_resLSTM_conf_2, 
            seed_1024 = conf_loss_pure_resLSTM_conf_3)

del conf_loss_pure_resLSTM_preds_1
del conf_loss_pure_resLSTM_preds_2
del conf_loss_pure_resLSTM_preds_3
del conf_loss_pure_resLSTM_conf_1
del conf_loss_pure_resLSTM_conf_2
del conf_loss_pure_resLSTM_conf_3
gc.collect()

# conf loss pao_model
print("Running conf loss pao_model inference...")
conf_loss_pao_model_preds_1, conf_loss_pao_model_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                      conf_loss_model_paths['conf_loss_pao_model_seed_7'],
                                                                                      actual_input_v2_rh_mc,
                                                                                      torch_input_v2_rh_mc,
                                                                                      out_scale_v2_rh_mc)

conf_loss_pao_model_preds_2, conf_loss_pao_model_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                        conf_loss_model_paths['conf_loss_pao_model_seed_43'],
                                                                                        actual_input_v2_rh_mc,
                                                                                        torch_input_v2_rh_mc,
                                                                                        out_scale_v2_rh_mc)

conf_loss_pao_model_preds_3, conf_loss_pao_model_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                        conf_loss_model_paths['conf_loss_pao_model_seed_1024'],
                                                                                        actual_input_v2_rh_mc,
                                                                                        torch_input_v2_rh_mc,
                                                                                        out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_pao_model_preds.npz'),
            seed_7 = conf_loss_pao_model_preds_1, 
            seed_43 = conf_loss_pao_model_preds_2, 
            seed_1024 = conf_loss_pao_model_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_pao_model_conf.npz'),
            seed_7 = conf_loss_pao_model_conf_1, 
            seed_43 = conf_loss_pao_model_conf_2, 
            seed_1024 = conf_loss_pao_model_conf_3)

del conf_loss_pao_model_preds_1
del conf_loss_pao_model_preds_2
del conf_loss_pao_model_preds_3
del conf_loss_pao_model_conf_1
del conf_loss_pao_model_conf_2
del conf_loss_pao_model_conf_3
gc.collect()

# conf loss convnext
print("Running conf loss convnext inference...")
conf_loss_convnext_preds_1, conf_loss_convnext_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                      conf_loss_model_paths['conf_loss_convnext_seed_7'],
                                                                                      actual_input_v2_rh_mc,
                                                                                      torch_input_v2_rh_mc,
                                                                                      out_scale_v2_rh_mc)  

conf_loss_convnext_preds_2, conf_loss_convnext_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                        conf_loss_model_paths['conf_loss_convnext_seed_43'],
                                                                                        actual_input_v2_rh_mc,
                                                                                        torch_input_v2_rh_mc,
                                                                                        out_scale_v2_rh_mc)

conf_loss_convnext_preds_3, conf_loss_convnext_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                        conf_loss_model_paths['conf_loss_convnext_seed_1024'],
                                                                                        actual_input_v2_rh_mc,
                                                                                        torch_input_v2_rh_mc,
                                                                                        out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_convnext_preds.npz'),
            seed_7 = conf_loss_convnext_preds_1, 
            seed_43 = conf_loss_convnext_preds_2, 
            seed_1024 = conf_loss_convnext_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_convnext_conf.npz'),
            seed_7 = conf_loss_convnext_conf_1, 
            seed_43 = conf_loss_convnext_conf_2, 
            seed_1024 = conf_loss_convnext_conf_3)

del conf_loss_convnext_preds_1
del conf_loss_convnext_preds_2
del conf_loss_convnext_preds_3
del conf_loss_convnext_conf_1
del conf_loss_convnext_conf_2
del conf_loss_convnext_conf_3
gc.collect()

# conf loss encdec_lstm
print("Running conf loss encdec_lstm inference...")
conf_loss_encdec_lstm_preds_1, conf_loss_encdec_lstm_conf_1 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                          conf_loss_model_paths['conf_loss_encdec_lstm_seed_7'],
                                                                                          actual_input_v2_rh_mc,
                                                                                          torch_input_v2_rh_mc,
                                                                                          out_scale_v2_rh_mc)

conf_loss_encdec_lstm_preds_2, conf_loss_encdec_lstm_conf_2 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                            conf_loss_model_paths['conf_loss_encdec_lstm_seed_43'],
                                                                                            actual_input_v2_rh_mc,
                                                                                            torch_input_v2_rh_mc,
                                                                                            out_scale_v2_rh_mc)

conf_loss_encdec_lstm_preds_3, conf_loss_encdec_lstm_conf_3 = inference_model_conf_loss(data_v2_rh_mc,
                                                                                            conf_loss_model_paths['conf_loss_encdec_lstm_seed_1024'],
                                                                                            actual_input_v2_rh_mc,
                                                                                            torch_input_v2_rh_mc,
                                                                                            out_scale_v2_rh_mc)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_encdec_lstm_preds.npz'),
            seed_7 = conf_loss_encdec_lstm_preds_1, 
            seed_43 = conf_loss_encdec_lstm_preds_2, 
            seed_1024 = conf_loss_encdec_lstm_preds_3)

np.savez(os.path.join(conf_loss_save_path, 'conf_loss_encdec_lstm_conf.npz'),
            seed_7 = conf_loss_encdec_lstm_conf_1, 
            seed_43 = conf_loss_encdec_lstm_conf_2, 
            seed_1024 = conf_loss_encdec_lstm_conf_3)

del conf_loss_encdec_lstm_preds_1
del conf_loss_encdec_lstm_preds_2
del conf_loss_encdec_lstm_preds_3
del conf_loss_encdec_lstm_conf_1
del conf_loss_encdec_lstm_conf_2
del conf_loss_encdec_lstm_conf_3
gc.collect()

# diff loss unet
print("Running diff loss unet inference...")
diff_loss_unet_preds_1 = inference_model(data_v2_rh_mc,
                                          diff_loss_model_paths['diff_loss_unet_seed_7'],
                                          actual_input_v2_rh_mc,
                                          torch_input_v2_rh_mc,
                                          out_scale_v2_rh_mc)

diff_loss_unet_preds_2 = inference_model(data_v2_rh_mc,
                                            diff_loss_model_paths['diff_loss_unet_seed_43'],
                                            actual_input_v2_rh_mc,
                                            torch_input_v2_rh_mc,
                                            out_scale_v2_rh_mc)

diff_loss_unet_preds_3 = inference_model(data_v2_rh_mc,
                                            diff_loss_model_paths['diff_loss_unet_seed_1024'],
                                            actual_input_v2_rh_mc,
                                            torch_input_v2_rh_mc,
                                            out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_unet_preds.npz'),
            seed_7 = diff_loss_unet_preds_1, 
            seed_43 = diff_loss_unet_preds_2, 
            seed_1024 = diff_loss_unet_preds_3)

del diff_loss_unet_preds_1
del diff_loss_unet_preds_2
del diff_loss_unet_preds_3
gc.collect()

# diff loss squeezeformer
print("Running diff loss squeezeformer inference...")
diff_loss_squeezeformer_preds_1 = inference_model(data_v2_rh_mc,
                                                   diff_loss_model_paths['diff_loss_squeezeformer_seed_7'],
                                                   actual_input_v2_rh_mc,
                                                   torch_input_v2_rh_mc,
                                                   out_scale_v2_rh_mc)

diff_loss_squeezeformer_preds_2 = inference_model(data_v2_rh_mc,
                                                    diff_loss_model_paths['diff_loss_squeezeformer_seed_43'],
                                                    actual_input_v2_rh_mc,
                                                    torch_input_v2_rh_mc,
                                                    out_scale_v2_rh_mc)

diff_loss_squeezeformer_preds_3 = inference_model(data_v2_rh_mc,
                                                    diff_loss_model_paths['diff_loss_squeezeformer_seed_1024'],
                                                    actual_input_v2_rh_mc,
                                                    torch_input_v2_rh_mc,
                                                    out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_squeezeformer_preds.npz'),
            seed_7 = diff_loss_squeezeformer_preds_1, 
            seed_43 = diff_loss_squeezeformer_preds_2, 
            seed_1024 = diff_loss_squeezeformer_preds_3)

del diff_loss_squeezeformer_preds_1
del diff_loss_squeezeformer_preds_2
del diff_loss_squeezeformer_preds_3
gc.collect()

# diff loss pure_resLSTM
print("Running diff loss pure_resLSTM inference...")
diff_loss_pure_resLSTM_preds_1 = inference_model(data_v2_rh_mc,
                                                  diff_loss_model_paths['diff_loss_pure_resLSTM_seed_7'],
                                                  actual_input_v2_rh_mc,
                                                  torch_input_v2_rh_mc,
                                                  out_scale_v2_rh_mc)

diff_loss_pure_resLSTM_preds_2 = inference_model(data_v2_rh_mc,
                                                    diff_loss_model_paths['diff_loss_pure_resLSTM_seed_43'],
                                                    actual_input_v2_rh_mc,
                                                    torch_input_v2_rh_mc,
                                                    out_scale_v2_rh_mc)

diff_loss_pure_resLSTM_preds_3 = inference_model(data_v2_rh_mc,
                                                    diff_loss_model_paths['diff_loss_pure_resLSTM_seed_1024'],
                                                    actual_input_v2_rh_mc,
                                                    torch_input_v2_rh_mc,
                                                    out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_pure_resLSTM_preds.npz'),
            seed_7 = diff_loss_pure_resLSTM_preds_1, 
            seed_43 = diff_loss_pure_resLSTM_preds_2, 
            seed_1024 = diff_loss_pure_resLSTM_preds_3)

del diff_loss_pure_resLSTM_preds_1
del diff_loss_pure_resLSTM_preds_2
del diff_loss_pure_resLSTM_preds_3
gc.collect()

# diff loss pao_model
print("Running diff loss pao_model inference...")
diff_loss_pao_model_preds_1 = inference_model(data_v2_rh_mc,
                                              diff_loss_model_paths['diff_loss_pao_model_seed_7'],
                                              actual_input_v2_rh_mc,
                                              torch_input_v2_rh_mc,
                                              out_scale_v2_rh_mc)

diff_loss_pao_model_preds_2 = inference_model(data_v2_rh_mc,
                                              diff_loss_model_paths['diff_loss_pao_model_seed_43'],
                                              actual_input_v2_rh_mc,
                                              torch_input_v2_rh_mc,
                                              out_scale_v2_rh_mc)

diff_loss_pao_model_preds_3 = inference_model(data_v2_rh_mc,
                                              diff_loss_model_paths['diff_loss_pao_model_seed_1024'],
                                              actual_input_v2_rh_mc,
                                              torch_input_v2_rh_mc,
                                              out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_pao_model_preds.npz'),
            seed_7 = diff_loss_pao_model_preds_1, 
            seed_43 = diff_loss_pao_model_preds_2, 
            seed_1024 = diff_loss_pao_model_preds_3)

del diff_loss_pao_model_preds_1
del diff_loss_pao_model_preds_2
del diff_loss_pao_model_preds_3
gc.collect()

# diff loss convnext
print("Running diff loss convnext inference...")
diff_loss_convnext_preds_1 = inference_model(data_v2_rh_mc,
                                             diff_loss_model_paths['diff_loss_convnext_seed_7'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

diff_loss_convnext_preds_2 = inference_model(data_v2_rh_mc,
                                             diff_loss_model_paths['diff_loss_convnext_seed_43'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

diff_loss_convnext_preds_3 = inference_model(data_v2_rh_mc,
                                             diff_loss_model_paths['diff_loss_convnext_seed_1024'],
                                             actual_input_v2_rh_mc,
                                             torch_input_v2_rh_mc,
                                             out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_convnext_preds.npz'),
            seed_7 = diff_loss_convnext_preds_1, 
            seed_43 = diff_loss_convnext_preds_2, 
            seed_1024 = diff_loss_convnext_preds_3)

del diff_loss_convnext_preds_1
del diff_loss_convnext_preds_2
del diff_loss_convnext_preds_3
gc.collect()

# diff loss encdec_lstm
print("Running diff loss encdec_lstm inference...")
diff_loss_encdec_lstm_preds_1 = inference_model(data_v2_rh_mc,
                                                diff_loss_model_paths['diff_loss_encdec_lstm_seed_7'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

diff_loss_encdec_lstm_preds_2 = inference_model(data_v2_rh_mc,
                                                diff_loss_model_paths['diff_loss_encdec_lstm_seed_43'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

diff_loss_encdec_lstm_preds_3 = inference_model(data_v2_rh_mc,
                                                diff_loss_model_paths['diff_loss_encdec_lstm_seed_1024'],
                                                actual_input_v2_rh_mc,
                                                torch_input_v2_rh_mc,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(diff_loss_save_path, 'diff_loss_encdec_lstm_preds.npz'),
            seed_7 = diff_loss_encdec_lstm_preds_1, 
            seed_43 = diff_loss_encdec_lstm_preds_2, 
            seed_1024 = diff_loss_encdec_lstm_preds_3)

del diff_loss_encdec_lstm_preds_1
del diff_loss_encdec_lstm_preds_2
del diff_loss_encdec_lstm_preds_3
gc.collect()

# multirep unet
print("Running multirep unet inference...")
multirep_unet_preds_1 = inference_model(data_v2_rh_mc,
                                        multirep_model_paths['multirep_unet_seed_7'],
                                        actual_input_multirep,
                                        torch_input_multirep,
                                        out_scale_v2_rh_mc)

multirep_unet_preds_2 = inference_model(data_v2_rh_mc,
                                        multirep_model_paths['multirep_unet_seed_43'],
                                        actual_input_multirep,
                                        torch_input_multirep,
                                        out_scale_v2_rh_mc)

multirep_unet_preds_3 = inference_model(data_v2_rh_mc,
                                        multirep_model_paths['multirep_unet_seed_1024'],
                                        actual_input_multirep,
                                        torch_input_multirep,
                                        out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_unet_preds.npz'),
         seed_7 = multirep_unet_preds_1, 
         seed_43 = multirep_unet_preds_2, 
         seed_1024 = multirep_unet_preds_3)

del multirep_unet_preds_1
del multirep_unet_preds_2
del multirep_unet_preds_3
gc.collect()

# multirep squeezeformer
print("Running multirep squeezeformer inference...")
multirep_squeezeformer_preds_1 = inference_model(data_v2_rh_mc,
                                                 multirep_model_paths['multirep_squeezeformer_seed_7'],
                                                 actual_input_multirep,
                                                 torch_input_multirep,
                                                 out_scale_v2_rh_mc)

multirep_squeezeformer_preds_2 = inference_model(data_v2_rh_mc,
                                                 multirep_model_paths['multirep_squeezeformer_seed_43'],
                                                 actual_input_multirep,
                                                 torch_input_multirep,
                                                 out_scale_v2_rh_mc)

multirep_squeezeformer_preds_3 = inference_model(data_v2_rh_mc,
                                                 multirep_model_paths['multirep_squeezeformer_seed_1024'],
                                                 actual_input_multirep,
                                                 torch_input_multirep,
                                                 out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_squeezeformer_preds.npz'),
            seed_7 = multirep_squeezeformer_preds_1, 
            seed_43 = multirep_squeezeformer_preds_2, 
            seed_1024 = multirep_squeezeformer_preds_3)

del multirep_squeezeformer_preds_1
del multirep_squeezeformer_preds_2
del multirep_squeezeformer_preds_3
gc.collect()

# multirep pure_resLSTM
print("Running multirep pure_resLSTM inference...")
multirep_pure_resLSTM_preds_1 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pure_resLSTM_seed_7'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_pure_resLSTM_preds_2 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pure_resLSTM_seed_43'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_pure_resLSTM_preds_3 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pure_resLSTM_seed_1024'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_pure_resLSTM_preds.npz'),
            seed_7 = multirep_pure_resLSTM_preds_1, 
            seed_43 = multirep_pure_resLSTM_preds_2, 
            seed_1024 = multirep_pure_resLSTM_preds_3)

del multirep_pure_resLSTM_preds_1
del multirep_pure_resLSTM_preds_2
del multirep_pure_resLSTM_preds_3
gc.collect()  

# multirep pao_model
print("Running multirep pao_model inference...")
multirep_pao_model_preds_1 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pao_model_seed_7'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_pao_model_preds_2 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pao_model_seed_43'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_pao_model_preds_3 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_pao_model_seed_1024'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_pao_model_preds.npz'),
            seed_7 = multirep_pao_model_preds_1, 
            seed_43 = multirep_pao_model_preds_2, 
            seed_1024 = multirep_pao_model_preds_3)

del multirep_pao_model_preds_1
del multirep_pao_model_preds_2
del multirep_pao_model_preds_3
gc.collect()

# multirep convnext
print("Running multirep convnext inference...")
multirep_convnext_preds_1 = inference_model(data_v2_rh_mc,
                                             multirep_model_paths['multirep_convnext_seed_7'],
                                             actual_input_multirep,
                                             torch_input_multirep,
                                             out_scale_v2_rh_mc)

multirep_convnext_preds_2 = inference_model(data_v2_rh_mc,
                                             multirep_model_paths['multirep_convnext_seed_43'],
                                             actual_input_multirep,
                                             torch_input_multirep,
                                             out_scale_v2_rh_mc)

multirep_convnext_preds_3 = inference_model(data_v2_rh_mc,
                                             multirep_model_paths['multirep_convnext_seed_1024'],
                                             actual_input_multirep,
                                             torch_input_multirep,
                                             out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_convnext_preds.npz'),
            seed_7 = multirep_convnext_preds_1, 
            seed_43 = multirep_convnext_preds_2, 
            seed_1024 = multirep_convnext_preds_3)

del multirep_convnext_preds_1
del multirep_convnext_preds_2
del multirep_convnext_preds_3
gc.collect()

# multirep encdec_lstm
print("Running multirep encdec_lstm inference...")
multirep_encdec_lstm_preds_1 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_encdec_lstm_seed_7'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_encdec_lstm_preds_2 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_encdec_lstm_seed_43'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

multirep_encdec_lstm_preds_3 = inference_model(data_v2_rh_mc,
                                                multirep_model_paths['multirep_encdec_lstm_seed_1024'],
                                                actual_input_multirep,
                                                torch_input_multirep,
                                                out_scale_v2_rh_mc)

np.savez(os.path.join(multirep_save_path, 'multirep_encdec_lstm_preds.npz'),
            seed_7 = multirep_encdec_lstm_preds_1, 
            seed_43 = multirep_encdec_lstm_preds_2, 
            seed_1024 = multirep_encdec_lstm_preds_3)

del multirep_encdec_lstm_preds_1
del multirep_encdec_lstm_preds_2
del multirep_encdec_lstm_preds_3
gc.collect()

# v6 unet
print("Running v6 unet inference...")
v6_unet_preds_1 = inference_model(data_v6,
                                    v6_model_paths['v6_unet_seed_7'],
                                    actual_input_v6,
                                    torch_input_v6,
                                    out_scale_v6)

v6_unet_preds_2 = inference_model(data_v6,
                                    v6_model_paths['v6_unet_seed_43'],
                                    actual_input_v6,
                                    torch_input_v6,
                                    out_scale_v6)

v6_unet_preds_3 = inference_model(data_v6,
                                    v6_model_paths['v6_unet_seed_1024'],
                                    actual_input_v6,
                                    torch_input_v6,
                                    out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_unet_preds.npz'),
            seed_7 = v6_unet_preds_1, 
            seed_43 = v6_unet_preds_2, 
            seed_1024 = v6_unet_preds_3)

del v6_unet_preds_1
del v6_unet_preds_2
del v6_unet_preds_3
gc.collect()

# v6 squeezeformer
print("Running v6 squeezeformer inference...")
v6_squeezeformer_preds_1 = inference_model(data_v6,
                                           v6_model_paths['v6_squeezeformer_seed_7'],
                                           actual_input_v6,
                                           torch_input_v6,
                                           out_scale_v6)

v6_squeezeformer_preds_2 = inference_model(data_v6,
                                           v6_model_paths['v6_squeezeformer_seed_43'],
                                           actual_input_v6,
                                           torch_input_v6,
                                           out_scale_v6)

v6_squeezeformer_preds_3 = inference_model(data_v6,
                                           v6_model_paths['v6_squeezeformer_seed_1024'],
                                           actual_input_v6,
                                           torch_input_v6,
                                           out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_squeezeformer_preds.npz'),
            seed_7 = v6_squeezeformer_preds_1, 
            seed_43 = v6_squeezeformer_preds_2, 
            seed_1024 = v6_squeezeformer_preds_3)

del v6_squeezeformer_preds_1
del v6_squeezeformer_preds_2
del v6_squeezeformer_preds_3
gc.collect()

# v6 pure_resLSTM
print("Running v6 pure_resLSTM inference...")
v6_pure_resLSTM_preds_1 = inference_model(data_v6,
                                            v6_model_paths['v6_pure_resLSTM_seed_7'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

v6_pure_resLSTM_preds_2 = inference_model(data_v6,
                                            v6_model_paths['v6_pure_resLSTM_seed_43'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

v6_pure_resLSTM_preds_3 = inference_model(data_v6,
                                            v6_model_paths['v6_pure_resLSTM_seed_1024'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_pure_resLSTM_preds.npz'),
            seed_7 = v6_pure_resLSTM_preds_1, 
            seed_43 = v6_pure_resLSTM_preds_2, 
            seed_1024 = v6_pure_resLSTM_preds_3)

del v6_pure_resLSTM_preds_1
del v6_pure_resLSTM_preds_2
del v6_pure_resLSTM_preds_3
gc.collect()

# v6 pao_model
print("Running v6 pao_model inference...")
v6_pao_model_preds_1 = inference_model(data_v6,
                                       v6_model_paths['v6_pao_model_seed_7'],
                                       actual_input_v6,
                                       torch_input_v6,
                                       out_scale_v6)

v6_pao_model_preds_2 = inference_model(data_v6,
                                       v6_model_paths['v6_pao_model_seed_43'],
                                       actual_input_v6,
                                       torch_input_v6,
                                       out_scale_v6)

v6_pao_model_preds_3 = inference_model(data_v6,
                                        v6_model_paths['v6_pao_model_seed_1024'],
                                        actual_input_v6,
                                        torch_input_v6,
                                        out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_pao_model_preds.npz'),
            seed_7 = v6_pao_model_preds_1, 
            seed_43 = v6_pao_model_preds_2, 
            seed_1024 = v6_pao_model_preds_3)

del v6_pao_model_preds_1
del v6_pao_model_preds_2
del v6_pao_model_preds_3
gc.collect()

# v6 convnext
print("Running v6 convnext inference...")
v6_convnext_preds_1 = inference_model(data_v6,
                                        v6_model_paths['v6_convnext_seed_7'],
                                        actual_input_v6,
                                        torch_input_v6,
                                        out_scale_v6)

v6_convnext_preds_2 = inference_model(data_v6,
                                        v6_model_paths['v6_convnext_seed_43'],
                                        actual_input_v6,
                                        torch_input_v6,
                                        out_scale_v6)

v6_convnext_preds_3 = inference_model(data_v6,
                                        v6_model_paths['v6_convnext_seed_1024'],
                                        actual_input_v6,
                                        torch_input_v6,
                                        out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_convnext_preds.npz'),
            seed_7 = v6_convnext_preds_1, 
            seed_43 = v6_convnext_preds_2, 
            seed_1024 = v6_convnext_preds_3)

del v6_convnext_preds_1
del v6_convnext_preds_2
del v6_convnext_preds_3
gc.collect()

# v6 encdec_lstm
print("Running v6 encdec_lstm inference...")
v6_encdec_lstm_preds_1 = inference_model(data_v6,
                                            v6_model_paths['v6_encdec_lstm_seed_7'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

v6_encdec_lstm_preds_2 = inference_model(data_v6,
                                            v6_model_paths['v6_encdec_lstm_seed_43'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

v6_encdec_lstm_preds_3 = inference_model(data_v6,
                                            v6_model_paths['v6_encdec_lstm_seed_1024'],
                                            actual_input_v6,
                                            torch_input_v6,
                                            out_scale_v6)

np.savez(os.path.join(v6_save_path, 'v6_encdec_lstm_preds.npz'),
            seed_7 = v6_encdec_lstm_preds_1, 
            seed_43 = v6_encdec_lstm_preds_2, 
            seed_1024 = v6_encdec_lstm_preds_3)

del v6_encdec_lstm_preds_1
del v6_encdec_lstm_preds_2
del v6_encdec_lstm_preds_3
gc.collect()

print('Finished inferencing all models!')