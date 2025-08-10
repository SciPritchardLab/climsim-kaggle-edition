import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
import os, gc, sys, glob, string, argparse
from tqdm import tqdm
import time
import itertools
import sys
import pickle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
grid_area = grid_info['area'].values
area_weight = grid_area/np.sum(grid_area)
level = grid_info.lev.values

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

actual_input_v2_rh_mc = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/actual_input.npy')
actual_target_v2_rh_mc = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/actual_target.npy')

actual_input_v6 = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/actual_input.npy')
actual_target_v6 = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/actual_target.npy')

assert np.array_equal(actual_target_v2_rh_mc, actual_target_v6)
actual_target = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/original_test_vars/actual_target_v2.npy')
del actual_target_v2_rh_mc
del actual_target_v6

assert np.array_equal(actual_input_v2_rh_mc[:,:,data_v2_rh_mc.ps_index], 
                      actual_input_v6[:,:,data_v6.ps_index])

surface_pressure = actual_input_v2_rh_mc[:, :, data_v2_rh_mc.ps_index]
hyam_component = (data_v2_rh_mc.hyam * data_v2_rh_mc.p0)[None,None,:]
hybm_component = data_v2_rh_mc.hybm[None,None,:] * surface_pressure[:,:,None]
pressures = hyam_component + hybm_component
pressures_binned = data_v2_rh_mc.zonal_bin_weight_3d(pressures)
lat_bin_mids = data_v2_rh_mc.lat_bin_mids
lats = data_v2_rh_mc.lats
lons = data_v2_rh_mc.lons

idx_p400_t10 = np.load('/pscratch/sd/z/zeyuanhu/hu_etal2024_data/microphysics_hourly/first_true_indices_p400_t10.npy')
for i in range(idx_p400_t10.shape[0]):
    for j in range(idx_p400_t10.shape[1]):
        idx_p400_t10[i,j] = level[int(idx_p400_t10[i,j])]

idx_p400_t10 = idx_p400_t10.mean(axis=0)
idx_p400_t10 = idx_p400_t10[np.newaxis,:]

idx_tropopause_zm = data_v2_rh_mc.zonal_bin_weight_2d(idx_p400_t10).flatten()

area_weight_dict = {
    'global': area_weight,
    'nh': np.where(lats > 30, area_weight, 0),
    'sh': np.where(lats < -30, area_weight, 0),
    'tropics': np.where((lats > -30) & (lats < 30), area_weight, 0)
}

lat_idx_dict = {
    '30S_30N': ((data_v2_rh_mc.lats < 30) & (data_v2_rh_mc.lats > -30))[None,:,None],
    '30N_60N': ((data_v2_rh_mc.lats < 60) & (data_v2_rh_mc.lats > 30))[None,:,None],
    '30S_60S': ((data_v2_rh_mc.lats < -30) & (data_v2_rh_mc.lats > -60))[None,:,None],
    '60N_90N': (data_v2_rh_mc.lats > 60)[None,:,None],
    '60S_90S': (data_v2_rh_mc.lats < -60)[None,:,None]
}

pressure_idx_dict = {
    'below_400hPa': pressures >= 400,
    'above_400hPa': pressures < 400
}

config_names = {
    'standard': 'Standard',
    'conf_loss': 'Confidence Loss',
    'diff_loss': 'Difference Loss',
    'multirep': 'Multirepresentation',
    'v6': 'Expanded Variable List'
}

model_names = {
    'unet': 'U-Net',
    'squeezeformer': 'Squeezeformer',
    'pure_resLSTM': 'Pure ResLSTM',
    'pao_model': 'Pao Model',
    'convnext': 'ConvNeXt',
    'encdec_lstm': 'Encoder-Decoder LSTM'
}

color_dict = {
    'unet': 'green',
    'squeezeformer': 'purple',
    'pure_resLSTM': 'blue',
    'pao_model': 'red',
    'convnext': 'gold',
    'encdec_lstm': 'orange',
}

color_dict_config = {
    'standard': 'blue',
    'conf_loss': 'cyan',
    'diff_loss': 'red',
    'multirep': 'orange',
    'v6': 'green'
}

offline_var_settings = {
    'DTPHYS': {'var_title': 'dT/dt', 'scaling': 1., 'unit': 'K/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':0},
    'DQ1PHYS': {'var_title': 'dQv/dt', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1e-6, 'vmin': -1e-6, 'var_index':60},
    'DQ2PHYS': {'var_title': 'dQl/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':120},
    'DQ3PHYS': {'var_title': 'dQi/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':180},
    'DUPHYS': {'var_title': 'dU/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':240},
    'DVPHYS': {'var_title': 'dV/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':300}
}

online_var_settings = {
    'T': {'var_title': 'Temperature', 'scaling': 1.0, 'unit': 'K', 'vmax': 5, 'vmin': -5},
    'Q': {'var_title': 'Specific Humidity', 'scaling': 1000.0, 'unit': 'g/kg', 'vmax': 1, 'vmin': -1},
    'U': {'var_title': 'Zonal Wind', 'scaling': 1.0, 'unit': 'm/s', 'vmax': 4, 'vmin': -4},
    'V': {'var_title': 'Meridional Wind', 'scaling': 1.0, 'unit': 'm/s', 'vmax': 4, 'vmin': -4},
    'CLDLIQ': {'var_title': 'Liquid Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 40, 'vmin': -40},
    'CLDICE': {'var_title': 'Ice Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 5, 'vmin': -5},
    'TOTCLD': {'var_title': 'Total Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 40, 'vmin': -40},
    'DTPHYS': {'var_title': 'Heating Tendency', 'scaling': 1., 'unit': 'K/s', 'vmax': 1.5e-5, 'vmin': -1.5e-5},
    'DQ1PHYS': {'var_title': 'Moistening Tendency', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1.2e-5, 'vmin': -1.2e-5},
    'DQ2PHYS': {'var_title': 'Liquid Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 0.0015, 'vmin': -0.0015},
    'DQ3PHYS': {'var_title': 'Ice Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 0.0015, 'vmin': -0.0015},
    'DQnPHYS': {'var_title': 'Liquid + Ice Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': .0015, 'vmin': -.0015},
    'DUPHYS': {'var_title': 'Zonal Wind Tendency', 'scaling': 1., 'unit': 'm/s²', 'vmax': 2.2e-6, 'vmin': -2.2e-6}
}

online_paths = {
    'standard': '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/standard/five_year_runs/',
    'conf_loss': '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/conf/five_year_runs/',
    'diff_loss': '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/diff/five_year_runs/',
    'multirep': '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/multirep/five_year_runs/',
    'v6': '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/v6/five_year_runs/'
}

seeds = ['seed_7', 'seed_43', 'seed_1024']
seed_numbers = [7, 43, 1024]

climsim3_figures_save_path_offline = '/global/cfs/cdirs/m4334/jerry/climsim3_figures/offline'
climsim3_figures_save_path_online = '/global/cfs/cdirs/m4334/jerry/climsim3_figures/online'

def ls(data_path = ""):
    return os.popen(" ".join(["ls", data_path])).read().splitlines()

def offline_area_time_mean_3d(arr):
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr)
    arr_zonal_time_mean = arr_zonal_mean.mean(axis = 0)
    arr_zonal_time_mean = xr.DataArray(arr_zonal_time_mean.T, dims = ['hybrid pressure (hPa)', 'latitude'], coords = {'hybrid pressure (hPa)':level, 'latitude': lat_bin_mids})
    return arr_zonal_time_mean

def online_area_time_mean_3d(ds, var):
    arr = ds[var].values[1:,:,:]
    arr_reshaped = np.transpose(arr, (0,2,1))
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr_reshaped)
    arr_zonal_time_mean = arr_zonal_mean.mean(axis = 0)
    arr_zonal_time_mean = xr.DataArray(arr_zonal_time_mean.T, dims = ['hybrid pressure (hPa)', 'latitude'], coords = {'hybrid pressure (hPa)':level, 'latitude': lat_bin_mids})
    return arr_zonal_time_mean

def area_mean(ds, var):
    arr = ds[var].values
    arr_reshaped = np.transpose(arr, (0,2,1))
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr_reshaped)
    return arr_zonal_mean

def zonal_diff(ds_sp, ds_nn, var):
    diff_zonal_mean = (area_mean(ds_nn, var) - area_mean(ds_sp, var)).mean(axis = 0)
    diff_zonal = xr.DataArray(diff_zonal_mean.T, dims = ['level', 'lat'], coords = {'level':level, 'lat': lat_bin_mids})
    return diff_zonal

def get_dp(ds):
    ps = ds['PS']
    p_interface = (ds['hyai'] * ds['P0'] + ds['hybi'] * ds['PS']).values
    if p_interface.shape[0] == 61:
        p_interface = np.swapaxes(p_interface, 0, 1)
    dp = p_interface[:,1:61,:] - p_interface[:,0:60,:]
    return dp

def get_tcp_mean(ds, area_weight):
    cld = ds['TOTCLD'].values
    dp = get_dp(ds)
    tcp = np.sum(cld*dp, axis = 1)/9.81
    tcp_mean = np.average(tcp, weights = area_weight, axis = 1)
    return tcp_mean

def read_mmf_online_data(num_years):
    assert num_years <= 5 and num_years >= 1
    years_regexp = '34567'[:num_years]
    ds_mmf_1 = xr.open_mfdataset(f'/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/h0/5year/mmf_ref/control_fullysp_jan_wmlio_r3.eam.h0.000[{years_regexp}]*.nc')
    ds_mmf_2 = xr.open_mfdataset(f'/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/h0/5year/mmf_b/control_fullysp_jan_wmlio_r3_b.eam.h0.000[{years_regexp}]*.nc')
    ds_mmf_1['DQnPHYS'] = ds_mmf_1['DQ2PHYS'] + ds_mmf_1['DQ3PHYS']
    ds_mmf_2['DQnPHYS'] = ds_mmf_2['DQ2PHYS'] + ds_mmf_2['DQ3PHYS']
    ds_mmf_1['TOTCLD'] = ds_mmf_1['CLDICE'] + ds_mmf_1['CLDLIQ']
    ds_mmf_2['TOTCLD'] = ds_mmf_2['CLDICE'] + ds_mmf_2['CLDLIQ']
    ds_mmf_1['PRECT'] = ds_mmf_1['PRECC'] + ds_mmf_1['PRECL']
    ds_mmf_2['PRECT'] = ds_mmf_2['PRECC'] + ds_mmf_2['PRECL']
    return ds_mmf_1, ds_mmf_2

def read_nn_online_data(config_name, model_name, seed, num_years):
    assert num_years <= 5 and num_years >= 1
    years_regexp = '34567'[:num_years]
    if config_name == 'standard':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_seed_{seed}', 'run', f'{model_name}_seed_{seed}.eam.h0.000[{years_regexp}]*.nc')
    elif config_name == 'conf_loss':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_conf_seed_{seed}', 'run', f'{model_name}_conf_seed_{seed}.eam.h0.000[{years_regexp}]*.nc')
    elif config_name == 'diff_loss':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_diff_seed_{seed}', 'run', f'{model_name}_diff_seed_{seed}.eam.h0.000[{years_regexp}]*.nc')
    elif config_name == 'multirep':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_multirep_seed_{seed}', 'run', f'{model_name}_multirep_seed_{seed}.eam.h0.000[{years_regexp}]*.nc')
    elif config_name == 'v6':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_v6_seed_{seed}', 'run', f'{model_name}_v6_seed_{seed}.eam.h0.000[{years_regexp}]*.nc')
    if len(ls(extract_path)) == 0:
        return None
    ds_nn = xr.open_mfdataset(extract_path)
    if len(ds_nn['time']) < 12 * num_years:
        return None
    ds_nn['DQnPHYS'] = ds_nn['DQ2PHYS'] + ds_nn['DQ3PHYS']
    ds_nn['TOTCLD'] = ds_nn['CLDICE'] + ds_nn['CLDLIQ']
    ds_nn['PRECT'] = ds_nn['PRECC'] + ds_nn['PRECL']
    return ds_nn

def read_nn_online_precip_data(config_name, model_name, seed, num_years):
    assert num_years <= 5 and num_years >= 1
    years_regexp = '34567'[:num_years]
    if config_name == 'standard':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_seed_{seed}', 'run', 'precip_dir', 'combined_precip.nc')
    elif config_name == 'conf_loss':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_conf_seed_{seed}', 'run', 'precip_dir', 'combined_precip.nc')
    elif config_name == 'diff_loss':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_diff_seed_{seed}', 'run', 'precip_dir', 'combined_precip.nc')
    elif config_name == 'multirep':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_multirep_seed_{seed}', 'run', 'precip_dir', 'combined_precip.nc')
    elif config_name == 'v6':
        extract_path = os.path.join(online_paths[config_name], f'{model_name}_v6_seed_{seed}', 'run', 'precip_dir', 'combined_precip.nc')
    if len(ls(extract_path)) == 0:
        return None
    ds_nn = xr.open_dataset(extract_path)
    if len(ds_nn['time']) < 365 * 24 * num_years:
        return None
    return ds_nn['PRECT']

def get_pressure_area_weights(ds, surface_type = None):
    ds_dp = get_dp(ds)
    ds_total_weight = ds_dp * area_weight[None, None, :]
    ds_total_weight = ds_total_weight.mean(axis = 0)
    ds_total_weight = ds_total_weight/ds_total_weight.sum()
    if surface_type is None:
        return ds_total_weight
    elif surface_type == 'land':
        land_area = ds['LANDFRAC'].values * grid_area[None, :]
        land_area_sums = np.array([[np.sum(land_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(land_area.shape[0])])
        land_area_divs = np.stack([np.divide(1, land_area_sums[:, bin_index], where=~(land_area_sums[:, bin_index] == 0), out=np.zeros_like(land_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        land_area_weighting = land_area * land_area_divs
        return land_area_weighting
    elif surface_type == 'ocean':
        ocean_area = ds['OCNFRAC'].values * grid_area[None, :]
        ocean_area_sums = np.array([[np.sum(ocean_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(ocean_area.shape[0])])
        ocean_area_divs = np.stack([np.divide(1, ocean_area_sums[:, bin_index], where=~(ocean_area_sums[:, bin_index] == 0), out=np.zeros_like(ocean_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        ocean_area_weighting = ocean_area * ocean_area_divs
        return ocean_area_weighting
    elif surface_type == 'ice':
        ice_area = ds['ICEFRAC'].values * grid_area[None, :]
        ice_area_sums = np.array([[np.sum(ice_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(ice_area.shape[0])])
        ice_area_divs = np.stack([np.divide(1, ice_area_sums[:, bin_index], where=~(ice_area_sums[:, bin_index] == 0), out=np.zeros_like(ice_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        ice_area_weighting = ice_area * ice_area_divs
        return ice_area_weighting
    else:
        raise ValueError("Invalid surface type. Choose from 'land', 'ocean', or 'ice'.")

def plot_online_global_rmse_model_comparison(config_name, num_years, show = False, save_path = None):
    months = np.arange(1, num_years * 12 + 1)
    ds_mmf_1, ds_mmf_2 = read_mmf_online_data(num_years)
    mmf_1_total_weight = get_pressure_area_weights(ds_mmf_1)
    ds_nn = {
        'unet': {seed_number: read_nn_online_data(config_name, 'unet', seed_number, num_years) for seed_number in seed_numbers},
        'squeezeformer': {seed_number: read_nn_online_data(config_name, 'squeezeformer', seed_number, num_years) for seed_number in seed_numbers},
        'pure_resLSTM': {seed_number: read_nn_online_data(config_name, 'pure_resLSTM', seed_number, num_years) for seed_number in seed_numbers},
        'pao_model': {seed_number: read_nn_online_data(config_name, 'pao_model', seed_number, num_years) for seed_number in seed_numbers},
        'convnext': {seed_number: read_nn_online_data(config_name, 'convnext', seed_number, num_years) for seed_number in seed_numbers},
        'encdec_lstm': {seed_number: read_nn_online_data(config_name, 'encdec_lstm', seed_number, num_years) for seed_number in seed_numbers}
    }
    variables = ['T', 'Q', 'CLDLIQ', 'CLDICE', 'U', 'V']
    ylim_upper = {
        'T': 5,
        'Q': 0.7,
        'CLDLIQ': 60,
        'CLDICE': 8,
        'U': 11,
        'V': 5
    }
    fig, axes = plt.subplots(2, 3, figsize=(8, 7), sharey=True, constrained_layout=True)  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten axes for easier iteration
    def load_nn_var_time_mean(ds_nn_xr, var, num_years):
        return_vals = np.full((data_v2_rh_mc.num_levels, data_v2_rh_mc.num_latlon), np.nan)
        if not ds_nn_xr or len(ds_nn_xr['time']) < num_years * 12:
            return return_vals
        else:
            return ds_nn_xr[var].mean(dim = 'time').values
    for ax, var in zip(axes, variables):
        ds_mmf_1_mean = ds_mmf_1[var].mean(dim = 'time').values * online_var_settings[var]['scaling']
        ds_mmf_2_mean = ds_mmf_2[var].mean(dim = 'time').values * online_var_settings[var]['scaling']
        mmf_rmse = np.sqrt(np.average((ds_mmf_2_mean - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight))
        mmf_rmse_global = np.sqrt(np.average((ds_mmf_2_mean - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight))
        ds_nn_rmse = {
            'unet': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['unet'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers]),
            'squeezeformer': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['squeezeformer'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers]),
            'pure_resLSTM': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['pure_resLSTM'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers]),
            'pao_model': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['pao_model'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers]),
            'convnext': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['convnext'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers]),
            'encdec_lstm': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['encdec_lstm'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, axis = 1, weights = area_weight)) for seed_number in seed_numbers])
        }
        ds_nn_rmse_global = {
            'unet': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['unet'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers]),
            'squeezeformer': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['squeezeformer'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers]),
            'pure_resLSTM': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['pure_resLSTM'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers]),
            'pao_model': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['pao_model'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers]),
            'convnext': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['convnext'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers]),
            'encdec_lstm': np.array([np.sqrt(np.average((load_nn_var_time_mean(ds_nn['encdec_lstm'][seed_number], var, num_years) * online_var_settings[var]['scaling'] - ds_mmf_1_mean) ** 2, weights = mmf_1_total_weight)) for seed_number in seed_numbers])
        }
        for model_name in model_names.keys():
            ax.fill_betweenx(
                level,
                np.nanmin(ds_nn_rmse[model_name], axis = 0),
                np.nanmax(ds_nn_rmse[model_name], axis = 0),
                color = color_dict[model_name],
                alpha=0.15
            )
        line_mmf, = ax.plot(mmf_rmse, level, label=f'{mmf_rmse_global:.2f}', linestyle='-.', color='black')
        if np.sum(np.isnan(ds_nn_rmse['unet'])) != np.prod(ds_nn_rmse['unet'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['unet'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_unet, = ax.plot(ds_nn_rmse['unet'][argidx,:], level, label = f"{ds_nn_rmse_global['unet'][argidx]:.2f}", color = color_dict['unet'], linestyle = '-.')
        else:
            line_unet, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['unet'], linestyle = '-.')
        if np.sum(np.isnan(ds_nn_rmse['squeezeformer'])) != np.prod(ds_nn_rmse['squeezeformer'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['squeezeformer'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_squeezeformer, = ax.plot(ds_nn_rmse['squeezeformer'][argidx,:], level, label = f"{ds_nn_rmse_global['squeezeformer'][argidx]:.2f}", color = color_dict['squeezeformer'], linestyle = '-.')
        else:
            line_squeezeformer, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['squeezeformer'], linestyle = '-.')
        if np.sum(np.isnan(ds_nn_rmse['pure_resLSTM'])) != np.prod(ds_nn_rmse['pure_resLSTM'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['pure_resLSTM'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_pure_resLSTM, = ax.plot(ds_nn_rmse['pure_resLSTM'][argidx,:], level, label = f"{ds_nn_rmse_global['pure_resLSTM'][argidx]:.2f}", color = color_dict['pure_resLSTM'], linestyle = '-.')
        else:
            line_pure_resLSTM, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['pure_resLSTM'], linestyle = '-.')
        if np.sum(np.isnan(ds_nn_rmse['pao_model'])) != np.prod(ds_nn_rmse['pao_model'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['pao_model'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_pao_model, = ax.plot(ds_nn_rmse['pao_model'][argidx,:], level, label = f"{ds_nn_rmse_global['pao_model'][argidx]:.2f}", color = color_dict['pao_model'], linestyle = '-.')
        else:
            line_pao_model, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['pao_model'], linestyle = '-.')
        if np.sum(np.isnan(ds_nn_rmse['convnext'])) != np.prod(ds_nn_rmse['convnext'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['convnext'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_convnext, = ax.plot(ds_nn_rmse['convnext'][argidx,:], level, label = f"{ds_nn_rmse_global['convnext'][argidx]:.2f}", color = color_dict['convnext'], linestyle = '-.')
        else:
            line_convnext, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['convnext'], linestyle = '-.')
        if np.sum(np.isnan(ds_nn_rmse['encdec_lstm'])) != np.prod(ds_nn_rmse['encdec_lstm'].shape):
            abs_diff = np.nanmean(np.abs(ds_nn_rmse['encdec_lstm'] - mmf_rmse), axis = 1)
            argidx = np.nanargmin(np.abs(abs_diff - np.nanmedian(abs_diff)))
            line_encdec_lstm, = ax.plot(ds_nn_rmse['encdec_lstm'][argidx,:], level, label = f"{ds_nn_rmse_global['encdec_lstm'][argidx]:.2f}", color = color_dict['encdec_lstm'], linestyle = '-.')
        else:
            line_encdec_lstm, = ax.plot(np.full(level.shape, np.nan), level, label = 'N/A', color = color_dict['encdec_lstm'], linestyle = '-.')

        ax.set_xlim(left = 0, right = ylim_upper[var])
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(f"{online_var_settings[var]['var_title']} ({online_var_settings[var]['unit']})", fontsize=14, loc='center')  # Add main title with subplot label
        ax.set_xlabel(f"{online_var_settings[var]['unit']}", fontsize=14)  # Keep unit in x-label
        ax.invert_yaxis()  # Reverse the y-axis
        ax.legend(fontsize=8, ncol = 2)

    handles = [line_mmf, line_unet, line_squeezeformer, line_pure_resLSTM, line_pao_model, line_convnext, line_encdec_lstm]
    labels = ['MMF2', model_names['unet'], model_names['squeezeformer'], model_names['pure_resLSTM'], model_names['pao_model'], model_names['convnext'], model_names['encdec_lstm']]

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.27, 0.5), title='Model')
    fig.suptitle(f'{num_years} Year Global Mean Root Mean Squared Error ({config_names[config_name]} configuration)', fontsize=16)
    # Set a shared y-label for the first column
    axes[0].set_ylabel('Hybrid pressure (hPa)', fontsize=14)
    axes[3].set_ylabel('Hybrid pressure (hPa)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.savefig('state_rmse_profiles_and_scalar.pdf', format='pdf', dpi=400, bbox_inches='tight')
    if save_path:
        plt.savefig(os.path.join(save_path, f'online_{num_years}_year_global_RMSE_model_comparison_{config_name}.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

years_to_try = [4,5]
for years in years_to_try:
    for config_name in config_names.keys():
        plot_online_global_rmse_model_comparison(config_name, years, show=False, save_path=os.path.join(climsim3_figures_save_path_online, 'online_global_rmse_model_comparison'))