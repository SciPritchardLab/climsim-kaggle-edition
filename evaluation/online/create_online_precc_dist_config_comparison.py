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

def plot_precc_dist_config_comparison(num_years, model_name, surface_type = 'global', show = False, save_path = None):
    assert surface_type in ['global', 'land', 'ocean', 'ice']
    ds_mmf_1, ds_mmf_2 = read_mmf_online_data(num_years)
    mmf_1_precc = ds_mmf_1['PRECT'].values
    nn_precc = {config_name: {seed_number: read_nn_online_data(config_name, model_name, seed_number, num_years) for seed_number in seed_numbers} for config_name in config_names.keys()}
    mmf_1_hourly_prect = xr.open_mfdataset('/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data_hourly/precip_hourly/mmf_ref/PRECT*nc')['PRECT'].sel(time = slice(None, str(num_years + 2).zfill(4))).values * 86400 * 1000
    mmf_1_hourly_prect_flat = mmf_1_hourly_prect.flatten()
    nn_hourly_prect = {config_name: {seed_number: read_nn_online_precip_data(config_name, model_name, seed_number, num_years) for seed_number in seed_numbers} for config_name in config_names.keys()}
    nn_hourly_prect = {config_name: {seed_number: nn_hourly_prect[config_name][seed_number].sel(time = slice(None, str(num_years + 2).zfill(4))).values * 86400 * 1000 if nn_hourly_prect[config_name][seed_number] is not None else []
                                    for seed_number in seed_numbers} for config_name in config_names.keys()}
    nn_hourly_prect = {config_name: {seed_number: nn_hourly_prect[config_name][seed_number] if len(nn_hourly_prect[config_name][seed_number]) == 365 * 24 * num_years else []
                                    for seed_number in seed_numbers} for config_name in config_names.keys()}
    if surface_type == 'global':
        prefix_str = 'Global'
        mmf_1_custom_weighting = None
        nn_custom_weighting = {config_name: {seed_number: None for seed_number in seed_numbers} for config_name in config_names.keys()}
        tile_area = data_v2_rh_mc.grid_info_area
        mmf_flat_area_weights = np.tile(tile_area, mmf_1_hourly_prect.shape[0])
    elif surface_type == 'land':
        prefix_str = 'Land'
        mmf_1_custom_weighting = get_pressure_area_weights(ds_mmf_1, surface_type = 'land')
        nn_custom_weighting = {config_name: {seed_number: get_pressure_area_weights(nn_precc[config_name][seed_number], surface_type = 'land') if nn_precc[config_name][seed_number] is not None else mmf_1_custom_weighting
                                                         for seed_number in seed_numbers} for config_name in config_names.keys()}
        tile_area = data_v2_rh_mc.grid_info_area * np.mean(ds_mmf_1['LANDFRAC'], axis = 0)
        mmf_flat_area_weights = np.tile(tile_area, mmf_1_hourly_prect.shape[0])
    elif surface_type == 'ocean':
        prefix_str = 'Ocean'
        mmf_1_custom_weighting = get_pressure_area_weights(ds_mmf_1, surface_type = 'ocean')
        nn_custom_weighting = {config_name: {seed_number: get_pressure_area_weights(nn_precc[config_name][seed_number], surface_type = 'ocean') if nn_precc[config_name][seed_number] is not None else mmf_1_custom_weighting
                                                         for seed_number in seed_numbers} for config_name in config_names.keys()}
        tile_area = data_v2_rh_mc.grid_info_area * np.mean(ds_mmf_1['OCNFRAC'], axis = 0)
        mmf_flat_area_weights = np.tile(tile_area, mmf_1_hourly_prect.shape[0])
    elif surface_type == 'ice':
        prefix_str = 'Ice'
        mmf_1_custom_weighting = get_pressure_area_weights(ds_mmf_1, surface_type = 'ice')
        nn_custom_weighting = {config_name: {seed_number: get_pressure_area_weights(nn_precc[config_name][seed_number], surface_type = 'ice') if nn_precc[config_name][seed_number] is not None else mmf_1_custom_weighting
                                                         for seed_number in seed_numbers} for config_name in config_names.keys()}
        tile_area = data_v2_rh_mc.grid_info_area * np.mean(ds_mmf_1['ICEFRAC'], axis = 0)
        mmf_flat_area_weights = np.tile(tile_area, mmf_1_hourly_prect.shape[0])
    mmf_1_precc_mean = np.mean(data_v2_rh_mc.zonal_bin_weight_2d(mmf_1_precc, custom_weighting = mmf_1_custom_weighting) * 86400 * 1000, axis = 0)
    nn_precc_mean = {config_name: {seed_number: np.mean(data_v2_rh_mc.zonal_bin_weight_2d(nn_precc[config_name][seed_number]['PRECT'].values, custom_weighting = nn_custom_weighting[config_name][seed_number]) * 86400 * 1000, axis = 0)
                if nn_precc[config_name][seed_number] is not None and nn_precc[config_name][seed_number]['PRECT'].shape[0] == 12 * num_years else np.full(lat_bin_mids.shape, np.nan) for seed_number in seed_numbers} for config_name in config_names.keys()}

    lat_ticks = [-60, -30, 0, 30, 60]
    lat_labels = ['60S', '30S', '0', '30N', '60N']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.7))
    ax1 = axes[0]
    ax1.plot(lat_bin_mids, mmf_1_precc_mean, label='MMF', color='black', linestyle='-')
    for config_name in config_names.keys():
        model_mean_arr = np.array([nn_precc_mean[config_name][seed_number] for seed_number in seed_numbers])
        if np.sum(np.isnan(model_mean_arr)) == model_mean_arr.size:
            continue
        ax1.fill_between(
            lat_bin_mids,
            np.nanmin(model_mean_arr, axis=0),
            np.nanmax(model_mean_arr, axis=0),
            color=color_dict_config[config_name],
            alpha=0.15
        )
        argidx = np.nanargmin(np.nanmean(np.abs(model_mean_arr - np.nanmedian(model_mean_arr, axis = 0)), axis=1))
        ax1.plot(lat_bin_mids, model_mean_arr[argidx, :], label = f"{config_names[config_name]}", color=color_dict_config[config_name], linestyle='--')

    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Precipitation (mm/day)')
    ax1.set_xticks(lat_ticks)
    ax1.set_xticklabels(lat_labels)
    ax1.set_title('(a) Mean Precipitation')
    # change fontsize of legend and make it two column
    handles, labels = ax1.get_legend_handles_labels()

    ax1.legend(fontsize='small', ncol=2)
    ax1.set_ylim(0, 8)
    ax1.set_xlim(-90,90)
    # Second plot: Weighted histogram of precipitation
    ax2 = axes[1]
    bins_lev = np.arange(-2,180,4)
    bin_centers = (bins_lev[:-1] + bins_lev[1:]) / 2

    def plot_histogram(ax, data_flat, weights, label, color, linestyle):
        hist, bins = np.histogram(data_flat, bins=bins_lev, weights=weights, density=True)
        ax.plot(bin_centers, hist, label=label, color=color, linestyle=linestyle, linewidth=2)

    plot_histogram(ax2, mmf_1_hourly_prect_flat, mmf_flat_area_weights, 'MMF', 'black', '-')
    for config_name in config_names.keys():
        if all(len(nn_hourly_prect[config_name][seed_number]) == 0 for seed_number in seed_numbers):
            continue
        model_mean_arr = np.array([nn_precc_mean[config_name][seed_number] for seed_number in seed_numbers])
        if np.sum(np.isnan(model_mean_arr)) == model_mean_arr.size:
            continue
        nn_prect_hist = np.array([np.histogram(np.array(nn_hourly_prect[config_name][seed_number]).flatten(),
                                    bins=bins_lev,
                                    weights=np.tile(tile_area, np.array(nn_hourly_prect[config_name][seed_number]).shape[0]),
                                    density=True)[0] for seed_number in seed_numbers])
        ax2.fill_between(
            bin_centers,
            np.nanmin(nn_prect_hist, axis=0),
            np.nanmax(nn_prect_hist, axis=0),
            color=color_dict_config[config_name],
            alpha=0.15
        )
        argidx = np.nanargmin(np.nanmean(np.abs(nn_prect_hist - np.nanmedian(nn_prect_hist, axis = 0)), axis=1))
        ax2.plot(bin_centers, nn_prect_hist[argidx, :], label = f"{config_names[config_name]}", color=color_dict_config[config_name], linestyle='--')

    ax2.set_yscale('log')
    ax2.set_xlabel('Precipitation (mm/day)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Histogram of Precipitation')
    ax2.set_ylim(1e-8,0.5)
    ax2.set_xlim(0,180)

    handles1 = handles[:4]
    labels1 = labels[:4]
    handles2 = handles[4:]
    labels2 = labels[4:]

    # Add legends to each subplot
    ax1.legend(handles1, labels1, loc='upper left')
    ax2.legend(handles2, labels2, loc='upper right')

    # Adjust layout
    # plt.tight_layout()
    # add space between suptitle
    plt.suptitle(f'{prefix_str} Precipitation {num_years}-Year Distribution ({model_names[model_name]})', fontsize=14.5)
    plt.subplots_adjust(top=0.85, wspace=0.3)  # Adjust the top space and width space between subplots
    
    # plt.savefig('precipitation_distribution_hist_nopruning_noclass.eps', format='eps', dpi=600)
    if save_path:
        plt.savefig(os.path.join(save_path, f'online_{num_years}_year_{surface_type}_precc_dist_config_comparison_{model_name}.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

years_to_try = [4,5]
surface_types = ['global', 'land', 'ocean', 'ice']
for years in years_to_try:
    for model_name in model_names.keys():
        for surface_type in surface_types:
            plot_precc_dist_config_comparison(num_years = years, model_name = model_name, surface_type=surface_type, show=False, save_path=os.path.join(climsim3_figures_save_path_online, 'online_precc_dist_config_comparison'))