import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
import matplotlib.lines as mlines
from matplotlib.transforms import blended_transform_factory
import os, gc, sys, glob, string, argparse
from tqdm import tqdm
import time
import string
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
    'DTPHYS': {'var_title': 'dT/dt', 'var_title_long': 'Heating', 'scaling': 1., 'unit': 'K/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':0},
    'DQ1PHYS': {'var_title': 'dQv/dt', 'var_title_long': 'Moistening', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1e-6, 'vmin': -1e-6, 'var_index':60},
    'DQ2PHYS': {'var_title': 'dQl/dt', 'var_title_long': 'Liquid Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':120},
    'DQ3PHYS': {'var_title': 'dQi/dt', 'var_title_long': 'Ice Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':180},
    'DUPHYS': {'var_title': 'dU/dt', 'var_title_long': 'Zonal Wind Tendency', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':240},
    'DVPHYS': {'var_title': 'dV/dt', 'var_title_long': 'Meridional Wind Tendency', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':300}
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

sypd_standard = [
    17.97901035,
    17.97901035,
    18.05589744,
    18.90966668,
    19.03178614,
    18.65975565,
    17.80983045,
    18.08638973,
    17.93128756,
    21.78946853,
    21.9847479,
    22.20038272,
    25.36398855,
    25.19805864,
    25.19227595,
    23.54557187,
    23.82794497,
    23.447502
]

sypd_conf_loss = [
    17.50453936,
    17.29971515,
    17.44196107,
    18.20260208,
    18.19053688,
    18.04921777,
    16.88591639,
    17.06639861,
    17.10229289,
    20.55525559,
    20.63385549,
    20.42521955,
    23.65465549,
    23.3813296,
    23.25625602,
    22.31773163,
    22.25777422,
    21.88176458
]

sypd_diff_loss =[
    16.97469344,
    16.92105472,
    17.16379509,
    18.99145235,
    19.24026684,
    18.69869139,
    17.75509967,
    16.87358759,
    16.66731672,
    20.77602886,
    19.70212994,
    19.22005471,
    21.00260056,
    25.47730606,
    25.47730606,
    24.0893883,
    23.99985625,
    23.78792837
]

sypd_multirep = [
    18.33255552,
    18.35861568,
    18.3448099,
    19.00295884,
    16.93296157,
    18.19204416,
    17.23723679,
    17.27385405,
    17.27657263,
    18.10913991,
    19.86397164,
    21.35810934,
    23.51060144,
    23.60014809,
    24.52549157,
    23.19360711,
    23.43123639,
    23.23164752
]

sypd_v6 = [
    18.21284028,
    18.31251632,
    18.41758393,
    6.894533848,
    8.95130655,
    7.890410959,
    17.84446786,
    17.79133537,
    17.73373234,
    21.39020294,
    17.90628317,
    21.40600448,
    24.47883654,
    24.13656852,
    24.3122588,
    23.34100196,
    23.5582043,
    23.5169076
]

pc_standard = [
    12975373,
    12975373,
    12975373,
    44785225,
    44785225,
    44785225,
    15395341,
    15395341,
    15395341,
    18876133,
    18876133,
    18876133,
    26805429,
    26805429,
    26805429,
    18582976,
    18582976,
    18582976
]

pc_conf_loss = [
    12980634,
    12980634,
    12980634,
    44811862,
    44811862,
    44811862,
    15402010,
    15402010,
    15402010,
    25407346,
    25407346,
    25407346,
    26839242,
    26839242,
    26839242,
    20723124,
    20723124,
    20723124
]

pc_diff_loss = [
    12975373,
    12975373,
    12975373,
    44785225,
    44785225,
    44785225,
    15395341,
    15395341,
    15395341,
    18876133,
    18876133,
    18876133,
    26805429,
    26805429,
    26805429,
    18582976,
    18582976,
    18582976
]

pc_multirep = [
    12981517,
    12981517,
    12981517,
    44791369,
    44791369,
    44791369,
    15428109,
    15428109,
    15428109,
    18880613,
    18880613,
    18880613,
    26811573,
    26811573,
    26811573,
    21004960,
    21004960,
    21004960
]

pc_v6 = [
    12981517,
    12981517,
    12981517,
    44791369,
    44791369,
    44791369,
    15428109,
    15428109,
    15428109,
    18880373,
    18880373,
    18880373,
    26811573,
    26811573,
    26811573,
    21004960,
    21004960,
    21004960
]

tt_standard = [
    5.272,
    5.895,
    5.902,
    32.986,
    32.867,
    33,
    9.863,
    8.708,
    8.733,
    8.106,
    8.108,
    8.141,
    13.254,
    13.253,
    12.297,
    5.352,
    5.425,
    5.378
]

tt_conf_loss = [
    5.988,
    5.989,
    5.986,
    33.0425,
    33.012,
    32.976,
    8.719,
    8.698,
    8.69,
    8.476,
    8.442,
    8.452,
    13.387,
    13.41,
    13.405,
    5.439,
    5.418,
    5.372
]

tt_diff_loss = [
    5.949,
    5.955,
    5.933,
    32.898,
    32.817,
    32.962,
    8.031,
    8.034,
    8.039,
    8.122,
    7.545,
    8.144,
    12.309,
    13.017,
    13.033,
    5.485,
    5.453,
    5.41
]

tt_multirep = [
    5.935,
    5.939,
    5.94,
    32.859,
    32.873,
    34.003,
    8.714,
    8.712,
    8.716,
    7.708,
    8.361,
    8.373,
    12.358,
    12.423,
    13.238,
    5.597,
    5.589,
    5.662
]

tt_v6 = [
    6.285,
    6.317,
    6.291,
    34.599,
    32.981,
    33.007,
    8.809,
    8.208,
    8.753,
    8.458,
    8.42,
    8.488,
    13.368,
    13.363,
    13.16,
    6.374,
    6.42,
    6.342
]

standard_sypd_dict = {}
conf_loss_sypd_dict = {}
diff_loss_sypd_dict = {}
multirep_sypd_dict = {}
v6_sypd_dict = {}

standard_pc_dict = {}
conf_loss_pc_dict = {}
diff_loss_pc_dict = {}
multirep_pc_dict = {}
v6_pc_dict = {}

standard_tt_dict = {}
conf_loss_tt_dict = {}
diff_loss_tt_dict = {}
multirep_tt_dict = {}
v6_tt_dict = {}

i = 0
for model_name in model_names.keys():
    for seed_number in seed_numbers:
        standard_sypd_dict[f"{model_name}_{seed_number}"] = sypd_standard[i]
        conf_loss_sypd_dict[f"{model_name}_{seed_number}"] = sypd_conf_loss[i]
        diff_loss_sypd_dict[f"{model_name}_{seed_number}"] = sypd_diff_loss[i]
        multirep_sypd_dict[f"{model_name}_{seed_number}"] = sypd_multirep[i]
        v6_sypd_dict[f"{model_name}_{seed_number}"] = sypd_v6[i]
        standard_pc_dict[f"{model_name}_{seed_number}"] = pc_standard[i]
        conf_loss_pc_dict[f"{model_name}_{seed_number}"] = pc_conf_loss[i]
        diff_loss_pc_dict[f"{model_name}_{seed_number}"] = pc_diff_loss[i]
        multirep_pc_dict[f"{model_name}_{seed_number}"] = pc_multirep[i]
        v6_pc_dict[f"{model_name}_{seed_number}"] = pc_v6[i]
        standard_tt_dict[f"{model_name}_{seed_number}"] = tt_standard[i]
        conf_loss_tt_dict[f"{model_name}_{seed_number}"] = tt_conf_loss[i]
        diff_loss_tt_dict[f"{model_name}_{seed_number}"] = tt_diff_loss[i]
        multirep_tt_dict[f"{model_name}_{seed_number}"] = tt_multirep[i]
        v6_tt_dict[f"{model_name}_{seed_number}"] = tt_v6[i]
        i += 1

sypd_dict = {
    'standard': standard_sypd_dict,
    'conf_loss': conf_loss_sypd_dict,
    'diff_loss': diff_loss_sypd_dict,
    'multirep': multirep_sypd_dict,
    'v6': v6_sypd_dict
}
pc_dict = {
    'standard': standard_pc_dict,
    'conf_loss': conf_loss_pc_dict,
    'diff_loss': diff_loss_pc_dict,
    'multirep': multirep_pc_dict,
    'v6': v6_pc_dict
}
tt_dict = {
    'standard': standard_tt_dict,
    'conf_loss': conf_loss_tt_dict,
    'diff_loss': diff_loss_tt_dict,
    'multirep': multirep_tt_dict,
    'v6': v6_tt_dict
}

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

def get_offline_precip_area_weights(nn_preds, precc_index = 363):
    nn_precc = nn_preds[:,:,precc_index] * 86400 * 1000
    no_precip_mask = nn_precc[:,:] == 0
    nn_active_precip = nn_precc[:,:][~no_precip_mask]
    nn_active_precip_quantiles = np.quantile(nn_active_precip, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    active_precip_1_10_percentile_mask = (nn_precc[:,:] > 0) & (nn_precc[:,:] <= nn_active_precip_quantiles[0])
    active_precip_10_20_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[0]) & (nn_precc[:,:] <= nn_active_precip_quantiles[1])
    active_precip_20_30_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[1]) & (nn_precc[:,:] <= nn_active_precip_quantiles[2])
    active_precip_30_40_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[2]) & (nn_precc[:,:] <= nn_active_precip_quantiles[3])
    active_precip_40_50_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[3]) & (nn_precc[:,:] <= nn_active_precip_quantiles[4])
    active_precip_50_60_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[4]) & (nn_precc[:,:] <= nn_active_precip_quantiles[5])
    active_precip_60_70_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[5]) & (nn_precc[:,:] <= nn_active_precip_quantiles[6])
    active_precip_70_80_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[6]) & (nn_precc[:,:] <= nn_active_precip_quantiles[7])
    active_precip_80_90_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[7]) & (nn_precc[:,:] <= nn_active_precip_quantiles[8])
    active_precip_90_100_percentile_mask = (nn_precc[:,:] > nn_active_precip_quantiles[8]) & (nn_precc[:,:] <= nn_active_precip_quantiles[9])
    precip_area_dict = {
        'no_precip': no_precip_mask * area_weight[None,:],
        'active_precip_1_10_percentile': active_precip_1_10_percentile_mask * area_weight[None,:],
        'active_precip_10_20_percentile': active_precip_10_20_percentile_mask * area_weight[None,:],
        'active_precip_20_30_percentile': active_precip_20_30_percentile_mask * area_weight[None,:],
        'active_precip_30_40_percentile': active_precip_30_40_percentile_mask * area_weight[None,:],
        'active_precip_40_50_percentile': active_precip_40_50_percentile_mask * area_weight[None,:],
        'active_precip_50_60_percentile': active_precip_50_60_percentile_mask * area_weight[None,:],
        'active_precip_60_70_percentile': active_precip_60_70_percentile_mask * area_weight[None,:],
        'active_precip_70_80_percentile': active_precip_70_80_percentile_mask * area_weight[None,:],
        'active_precip_80_90_percentile': active_precip_80_90_percentile_mask * area_weight[None,:],
        'active_precip_90_100_percentile': active_precip_90_100_percentile_mask * area_weight[None,:]
    }
    precip_area_divs = {key: np.divide(np.ones_like(precip_area_dict[key]), np.sum(precip_area_dict[key]), out = np.zeros_like(precip_area_dict[key]), where = (precip_area_dict[key] != 0)) for key in precip_area_dict.keys()}
    for key in precip_area_dict.keys():
        precip_area_dict[key] = (precip_area_dict[key] * precip_area_divs[key])[:,:,None]
    return precip_area_dict

precip_percentile_labels = {
    'no_precip': 'No Precipitation',
    'active_precip_1_10_percentile': '1-10th percentile',
    'active_precip_10_20_percentile': '10-20th percentile',
    'active_precip_20_30_percentile': '20-30th percentile',
    'active_precip_30_40_percentile': '30-40th percentile',
    'active_precip_40_50_percentile': '40-50th percentile',
    'active_precip_50_60_percentile': '50-60th percentile',
    'active_precip_60_70_percentile': '60-70th percentile',
    'active_precip_70_80_percentile': '70-80th percentile',
    'active_precip_80_90_percentile': '80-90th percentile',
    'active_precip_90_100_percentile': '90-100th percentile'
}

standard_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/standard/'
conf_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/conf_loss/'
diff_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/diff_loss/'
multirep_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/multirep/'
v6_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/test_preds/'

def load_seed_data(save_path, npz_file, seed_key):
    with np.load(os.path.join(save_path, npz_file)) as data:
        return data[seed_key]

print('loading standard preds')
standard_preds = {
    'unet': lambda seed_key: load_seed_data(standard_save_path, 'standard_unet_preds.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(standard_save_path, 'standard_squeezeformer_preds.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(standard_save_path, 'standard_pure_resLSTM_preds.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(standard_save_path, 'standard_pao_model_preds.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(standard_save_path, 'standard_convnext_preds.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(standard_save_path, 'standard_encdec_lstm_preds.npz', seed_key)
}

print('loading conf loss preds')
conf_loss_preds = {
    'unet': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_unet_preds.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_squeezeformer_preds.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_pure_resLSTM_preds.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_pao_model_preds.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_convnext_preds.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_encdec_lstm_preds.npz', seed_key)
}

print('loading conf loss conf')
conf_loss_conf = {
    'unet': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_unet_conf.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_squeezeformer_conf.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_pure_resLSTM_conf.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_pao_model_conf.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_convnext_conf.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(conf_loss_save_path, 'conf_loss_encdec_lstm_conf.npz', seed_key)
}

print('loading diff loss preds')
diff_loss_preds = {
    'unet': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_unet_preds.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_squeezeformer_preds.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_pure_resLSTM_preds.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_pao_model_preds.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_convnext_preds.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(diff_loss_save_path, 'diff_loss_encdec_lstm_preds.npz', seed_key)
}

print('loading multirep preds')
multirep_preds = {
    'unet': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_unet_preds.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_squeezeformer_preds.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_pure_resLSTM_preds.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_pao_model_preds.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_convnext_preds.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(multirep_save_path, 'multirep_encdec_lstm_preds.npz', seed_key)
}

print('loading v6 preds')
v6_preds = {
    'unet': lambda seed_key: load_seed_data(v6_save_path, 'v6_unet_preds.npz', seed_key),
    'squeezeformer': lambda seed_key: load_seed_data(v6_save_path, 'v6_squeezeformer_preds.npz', seed_key),
    'pure_resLSTM': lambda seed_key: load_seed_data(v6_save_path, 'v6_pure_resLSTM_preds.npz', seed_key),
    'pao_model': lambda seed_key: load_seed_data(v6_save_path, 'v6_pao_model_preds.npz', seed_key),
    'convnext': lambda seed_key: load_seed_data(v6_save_path, 'v6_convnext_preds.npz', seed_key),
    'encdec_lstm': lambda seed_key: load_seed_data(v6_save_path, 'v6_encdec_lstm_preds.npz', seed_key)
}

config_preds = {
    'standard': standard_preds,
    'conf_loss': conf_loss_preds,
    'diff_loss': diff_loss_preds,
    'multirep': multirep_preds,
    'v6': v6_preds
}

print('loading Kaggle preds')
kaggle_check_path = '/pscratch/sd/j/jerrylin/kaggle_check'

greysnow = np.load(os.path.join(kaggle_check_path, 'prds.npy'))
greysnow = greysnow.reshape(-1, data_v2_rh_mc.num_latlon, 368)

adam = pd.read_parquet(os.path.join(kaggle_check_path, 'final_blend_v10.parquet'))
adam = adam.iloc[:,1:].values
adam = adam.reshape(-1, data_v2_rh_mc.num_latlon, 368)

print(greysnow.shape, adam.shape)

def xarray_minmax_magnitude(data_array, minmax = None, dim=None):
    assert minmax in ['min', 'max'], "minmax must be either 'min' or 'max'"
    # Get indices of minimum or maximum absolute values
    if minmax == 'min':
        idx = np.abs(data_array).argmin(dim=dim)
    else:
        idx = np.abs(data_array).argmax(dim=dim)
    return data_array.isel({dim: idx})

def get_offline_bias_minmax(config_name, var, show = True, save_path = None):
    beg_index = offline_var_settings[var]['var_index']
    end_index = beg_index + 60
    nn_preds = {model_name: np.mean(np.array([config_preds[config_name][model_name](seed) for seed in seeds]), axis = 0) for model_name in model_names.keys()}
    moistening_diffs = {model_name: (nn_preds[model_name][:,:,beg_index:end_index] - actual_target[:,:,beg_index:end_index]) * offline_var_settings[var]['scaling'] for model_name in model_names.keys()}
    offline_precip_area_weights = {model_name: get_offline_precip_area_weights(nn_preds[model_name]) for model_name in model_names.keys()}
    moistening_diffs_zonal_mean = {model_name: offline_area_time_mean_3d((nn_preds[model_name][:,:,beg_index:end_index] - actual_target[:,:,beg_index:end_index])) * offline_var_settings[var]['scaling'] for model_name in model_names.keys()}
    combined_zonal_mean_diffs = xr.concat([moistening_diffs_zonal_mean[model_name] for model_name in model_names.keys()], dim='ensemble')
    combined_zonal_mean_diffs_min = xarray_minmax_magnitude(combined_zonal_mean_diffs, minmax='min', dim='ensemble')
    combined_zonal_mean_diffs_max = xarray_minmax_magnitude(combined_zonal_mean_diffs, minmax='max', dim='ensemble')

    cmap_diffs = 'RdBu_r'
    cmap_range = 'Reds'
    cmap_binning = matplotlib.colormaps['viridis']
    colors = cmap_binning(np.linspace(0, 1, 11))
    alphas = np.linspace(0.1, 1, 11)

    plotted_artists = {}

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    labels = [f"({letter})" for letter in string.ascii_lowercase[:4]]
    latitude_ticks = [-60, -30, 0, 30, 60]
    latitude_labels = ['60S', '30S', '0', '30N', '60N']

    plotted_artists['minimum_zonal'] = combined_zonal_mean_diffs_min.plot(ax=axs[0,0], add_colorbar=False, cmap='RdBu_r', vmin=offline_var_settings[var]['vmin'], vmax=offline_var_settings[var]['vmax'])
    axs[0,0].set_title(f'{labels[0]} Minimum Magnitude Zonal Mean Bias', fontsize = 14)
    axs[0,0].invert_yaxis()
    axs[0,0].set_xlabel('')
    plotted_artists['maximum_zonal'] = combined_zonal_mean_diffs_max.plot(ax=axs[1,0], add_colorbar=False, cmap='RdBu_r', vmin=offline_var_settings[var]['vmin'], vmax=offline_var_settings[var]['vmax'])
    axs[1,0].set_title(f'{labels[1]} Maximum Magnitude Zonal Mean Bias', fontsize = 14)
    axs[1,0].invert_yaxis()

    img = plotted_artists['minimum_zonal']
    cbar = fig.colorbar(img, ax = axs[0,0], orientation='vertical', fraction=0.046, pad=0.04)
    img = plotted_artists['maximum_zonal']
    cbar = fig.colorbar(img, ax = axs[1,0], orientation='vertical', fraction=0.046, pad=0.04)

    handles = []
    legend_labels = []
    for precip_idx, precip_key in enumerate(precip_percentile_labels.keys()):
        precip_arr = np.stack([np.sum(np.sum(moistening_diffs[model_name] * offline_precip_area_weights[model_name][precip_key], axis=1), axis=0) for model_name in model_names.keys()])
        line, = axs[0,1].plot(precip_arr[np.abs(precip_arr).argmin(axis=0), np.arange(60)], level, label=precip_percentile_labels[precip_key], linestyle='-', color=colors[precip_idx], alpha=alphas[precip_idx])
        axs[1,1].plot(precip_arr[np.abs(precip_arr).argmax(axis=0), np.arange(60)], level, linestyle='-', color=colors[precip_idx], alpha=alphas[precip_idx])
        handles.append(line)
        legend_labels.append(precip_percentile_labels[precip_key])
    axs[0,1].invert_yaxis()
    axs[0,1].set_title(f'{labels[2]} Minimum Magnitude Binned Bias', fontsize = 14)
    axs[0,1].grid(True)
    axs[0,1].set_xlim(offline_var_settings[var]['vmin']*1.4, offline_var_settings[var]['vmax']*1.4)
    axs[1,1].invert_yaxis()
    axs[1,1].set_xlabel(offline_var_settings[var]['unit'])
    axs[1,1].set_title(f'{labels[3]} Maximum Magnitude Binned Bias', fontsize = 14)
    axs[1,1].grid(True)
    axs[1,1].set_xlim(offline_var_settings[var]['vmin']*1.4, offline_var_settings[var]['vmax']*1.4)

    split_point = 6
    axs[0,1].legend(handles[:split_point], legend_labels[:split_point], loc='upper right', fontsize=9, ncol=1)
    axs[1,1].legend(handles[split_point:], legend_labels[split_point:], loc='upper right', fontsize=9, ncol=1)

    plt.suptitle(f'Minimum and Maximum Magnitude Offline {offline_var_settings[var]["var_title_long"]} Bias ({config_names[config_name]})')
    if save_path:
        plt.savefig(os.path.join(save_path, f'offline_{var}_bias_minmax_{config_name}.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

save_path = os.path.join(climsim3_figures_save_path_offline, 'offline_bias_minmax')
for config_name in config_names.keys():
    if config_name != 'v6':
        continue
    for var in offline_var_settings.keys():
        print(f'Processing {config_name} {var}')
        get_offline_bias_minmax(config_name, var, show = False, save_path = save_path)

print('I am done!!!!')