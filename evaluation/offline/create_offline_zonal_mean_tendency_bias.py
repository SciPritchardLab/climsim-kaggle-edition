import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
import torch
import os, gc, sys, glob, string, argparse
from tqdm import tqdm
import itertools
import sys
import pickle
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

grid_area = grid_info['area'].values[None,:,None]
pressure_interface = (grid_info['hyai'].values * grid_info['P0'].values)[None,None,:] + (surface_pressure[:,:,None] * grid_info['hybi'].values[None,None,:])
dp = pressure_interface[:,:,1:61] - pressure_interface[:,:,0:60]
total_weight = grid_area * dp

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

standard_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/standard/'
conf_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/conf_loss/'
diff_loss_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/diff_loss/'
multirep_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_preds/multirep/'
v6_save_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v6/test_set/test_preds/'
offline_save_path = '/global/cfs/cdirs/m4334/jerry/climsim3_offline_results'

def load_seed_data(save_path, npz_file, seed_key):
    with np.load(os.path.join(save_path, npz_file)) as data:
        return data[seed_key]

seeds = ['seed_7', 'seed_43', 'seed_1024']

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

config_names = {
    'standard': 'Standard',
    'conf_loss': 'Confidence Loss',
    'diff_loss': 'Difference Loss',
    'multirep': 'Multirepresentation',
    'v6': 'v6 variable list'
}

model_names = {
    'unet': 'U-Net',
    'squeezeformer': 'Squeezeformer',
    'pure_resLSTM': 'Pure ResLSTM',
    'pao_model': 'Pao Model',
    'convnext': 'ConvNeXt',
    'encdec_lstm': 'Encoder-Decoder LSTM'
}

def area_time_mean_3d(arr):
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr)
    arr_zonal_time_mean = arr_zonal_mean.mean(axis = 0)
    arr_zonal_time_mean = xr.DataArray(arr_zonal_time_mean.T, dims = ['hybrid pressure (hPa)', 'latitude'], coords = {'hybrid pressure (hPa)':level, 'latitude': lat_bin_mids})
    return arr_zonal_time_mean

def plot_zonal_mean_tendency_bias(config_name, model_name, seed, actual_target):
    nn_preds = config_preds[config_name][model_name](seed)
    nn_diff = nn_preds - actual_target
    diff_dTdt = nn_diff[:,:,0:60]
    diff_dQvdt = nn_diff[:,:,60:120]
    diff_dQldt = nn_diff[:,:,120:180]
    diff_dQidt = nn_diff[:,:,180:240]
    diff_dUdt = nn_diff[:,:,240:300]
    diff_dVdt = nn_diff[:,:,300:360]

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 17))
    # Generate the panel labels
    labels = [f"({letter})" for letter in string.ascii_lowercase[:6]]
    latitude_ticks = [-60, -30, 0, 30, 60]
    latitude_labels = ['60S', '30S', '0', '30N', '60N']
    # Loop through each variable and its corresponding subplot row
    var_settings = {
        'DTPHYS': {'var_title': 'dT/dt', 'scaling': 1., 'unit': 'K/s', 'vmax': 5e-7, 'vmin': -5e-7},
        'DQ1PHYS': {'var_title': 'dQv/dt', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1e-6, 'vmin': -1e-6},
        'DQ2PHYS': {'var_title': 'dQl/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3},
        'DQ3PHYS': {'var_title': 'dQi/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3},
        'DUPHYS': {'var_title': 'dU/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7},
        'DVPHYS': {'var_title': 'dV/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7}
    }

    nn_diff_DTPHYS = var_settings['DTPHYS']['scaling'] * area_time_mean_3d(diff_dTdt)
    nn_diff_DQ1PHYS = var_settings['DQ1PHYS']['scaling'] * area_time_mean_3d(diff_dQvdt)
    nn_diff_DQ2PHYS = var_settings['DQ2PHYS']['scaling'] * area_time_mean_3d(diff_dQldt)
    nn_diff_DQ3PHYS = var_settings['DQ3PHYS']['scaling'] * area_time_mean_3d(diff_dQidt)
    nn_diff_DUPHYS = var_settings['DUPHYS']['scaling'] * area_time_mean_3d(diff_dUdt)
    nn_diff_DVPHYS = var_settings['DVPHYS']['scaling'] * area_time_mean_3d(diff_dVdt)

    plotted_artists = {}

    plotted_artists['DTPHYS'] = nn_diff_DTPHYS.plot(ax=axs[0,0], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DTPHYS']['vmin'], vmax=var_settings['DTPHYS']['vmax'])
    axs[0,0].set_title(f'{labels[0]} dT/dt Bias ({var_settings["DTPHYS"]["unit"]})')
    axs[0,0].invert_yaxis()
    axs[0,0].set_xlabel('Latitude')

    plotted_artists['DQ1PHYS'] = nn_diff_DQ1PHYS.plot(ax=axs[0,1], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DQ1PHYS']['vmin'], vmax=var_settings['DQ1PHYS']['vmax'])
    axs[0,1].set_title(f'{labels[1]} dQv/dt Bias ({var_settings["DQ1PHYS"]["unit"]})')
    axs[0,1].invert_yaxis()
    axs[0,1].set_xlabel('Latitude')

    plotted_artists['DQ2PHYS'] = nn_diff_DQ2PHYS.plot(ax=axs[1,0], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DQ2PHYS']['vmin'], vmax=var_settings['DQ2PHYS']['vmax'])
    axs[1,0].set_title(f'{labels[2]} dQl/dt Bias ({var_settings["DQ2PHYS"]["unit"]})')
    axs[1,0].invert_yaxis()
    axs[1,0].set_xlabel('Latitude')

    plotted_artists['DQ3PHYS'] = nn_diff_DQ3PHYS.plot(ax=axs[1,1], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DQ3PHYS']['vmin'], vmax=var_settings['DQ3PHYS']['vmax'])
    axs[1,1].set_title(f'{labels[3]} dQi/dt Bias ({var_settings["DQ3PHYS"]["unit"]})')
    axs[1,1].invert_yaxis()
    axs[1,1].set_xlabel('Latitude')

    plotted_artists['DUPHYS'] = nn_diff_DUPHYS.plot(ax=axs[2,0], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DUPHYS']['vmin'], vmax=var_settings['DUPHYS']['vmax'])
    axs[2,0].set_title(f'{labels[4]} dU/dt Bias ({var_settings["DUPHYS"]["unit"]})')
    axs[2,0].invert_yaxis()
    axs[2,0].set_xlabel('Latitude')

    plotted_artists['DVPHYS'] = nn_diff_DVPHYS.plot(ax=axs[2,1], add_colorbar=False, cmap='RdBu_r', vmin=var_settings['DVPHYS']['vmin'], vmax=var_settings['DVPHYS']['vmax'])
    axs[2,1].set_title(f'{labels[5]} dV/dt Bias ({var_settings["DVPHYS"]["unit"]})')
    axs[2,1].invert_yaxis()
    axs[2,1].set_xlabel('Latitude')

    # add a colorbar to each subplot

    var_order = ['DTPHYS', 'DQ1PHYS', 'DQ2PHYS', 'DQ3PHYS', 'DUPHYS', 'DVPHYS']
    for ax, var_key in zip(axs.flat, var_order):
        img = plotted_artists[var_key]  # Use the stored artist
        if img is not None: # Check if artist exists
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            if var_key in ['DQ2PHYS', 'DQ3PHYS']:
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0e}"))
        else:
            print(f"Warning: No artist found for variable {var_key} to create colorbar.")
    
    for ax in axs.flat:
        ax.set_xticks(latitude_ticks)  # Set the positions for the ticks
        ax.set_xticklabels(latitude_labels)  # Set the custom text labels

    plt.suptitle(f'Offline Tendency Biases for {model_names[model_name]} {config_names[config_name]} (seed {seed[5:]})', fontsize=14, x = .6, y = .95)
    plt.subplots_adjust(right=1, top=.9)
    plt.savefig(f'/pscratch/sd/j/jerrylin/climsim3_figures/offline_results/zonal_mean_biases/{model_name}/{model_name}_{config_name}_{seed}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

plot_zonal_mean_tendency_bias('standard', 'unet', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'unet', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'unet', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('standard', 'squeezeformer', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'squeezeformer', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'squeezeformer', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pure_resLSTM', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pure_resLSTM', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pure_resLSTM', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pao_model', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pao_model', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'pao_model', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('standard', 'convnext', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'convnext', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'convnext', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('standard', 'encdec_lstm', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('standard', 'encdec_lstm', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('standard', 'encdec_lstm', 'seed_1024', actual_target)

plot_zonal_mean_tendency_bias('conf_loss', 'unet', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'unet', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'unet', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'squeezeformer', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'squeezeformer', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'squeezeformer', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pure_resLSTM', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pure_resLSTM', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pure_resLSTM', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pao_model', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pao_model', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'pao_model', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'convnext', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'convnext', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'convnext', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'encdec_lstm', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'encdec_lstm', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('conf_loss', 'encdec_lstm', 'seed_1024', actual_target)

plot_zonal_mean_tendency_bias('diff_loss', 'unet', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'unet', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'unet', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'squeezeformer', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'squeezeformer', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'squeezeformer', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pure_resLSTM', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pure_resLSTM', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pure_resLSTM', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pao_model', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pao_model', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'pao_model', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'convnext', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'convnext', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'convnext', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'encdec_lstm', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'encdec_lstm', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('diff_loss', 'encdec_lstm', 'seed_1024', actual_target)

plot_zonal_mean_tendency_bias('multirep', 'unet', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'unet', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'unet', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'squeezeformer', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'squeezeformer', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'squeezeformer', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pure_resLSTM', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pure_resLSTM', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pure_resLSTM', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pao_model', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pao_model', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'pao_model', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'convnext', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'convnext', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'convnext', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'encdec_lstm', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'encdec_lstm', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('multirep', 'encdec_lstm', 'seed_1024', actual_target)

plot_zonal_mean_tendency_bias('v6', 'unet', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'unet', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'unet', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('v6', 'squeezeformer', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'squeezeformer', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'squeezeformer', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pure_resLSTM', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pure_resLSTM', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pure_resLSTM', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pao_model', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pao_model', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'pao_model', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('v6', 'convnext', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'convnext', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'convnext', 'seed_1024', actual_target)
plot_zonal_mean_tendency_bias('v6', 'encdec_lstm', 'seed_7', actual_target)
plot_zonal_mean_tendency_bias('v6', 'encdec_lstm', 'seed_43', actual_target)
plot_zonal_mean_tendency_bias('v6', 'encdec_lstm', 'seed_1024', actual_target)