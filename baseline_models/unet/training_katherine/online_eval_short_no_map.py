import xarray as xr
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plb
import matplotlib.image as imag
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, gc, glob
from tqdm import tqdm
import shutil
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from climsim_utils.data_utils import *
from moviepy.editor import *
from sklearn.metrics import r2_score
import psutil
import time
import argparse

def make_pressure_weights(grid_info, ps):
    pressure_grid_p1 = np.array(grid_info['P0']*grid_info['hyai'])[np.newaxis,:,np.newaxis]
    pressure_grid_p2 = grid_info['hybi'].values[np.newaxis,:,np.newaxis] * ps.values[:,np.newaxis,:]
    pressure_grid = pressure_grid_p1 + pressure_grid_p2
    dp = pressure_grid[:,1:61,:] - pressure_grid[:,0:60,:]
    pressure_weights = dp/np.sum(dp, axis = 1)[:, np.newaxis, :]
    return pressure_weights

def main(mmf_path, nn_path, save_path, var, max_day, \
         grid_path, input_mean_file, input_max_file, input_min_file, \
         output_scale_file, lbd_qn_file):

    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_mean_file)
    input_max = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_max_file)
    input_min = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_min_file)
    output_scale = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_file)
    lbd_qn = np.loadtxt('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + lbd_qn_file, delimiter = ',')

    data = data_utils(grid_info = grid_info, 
                    input_mean = input_mean, 
                    input_max = input_max, 
                    input_min = input_min, 
                    output_scale = output_scale,
                    qinput_log = False,
                    normalize = False)
    data.set_to_v2_rh_mc_vars()

    input_sub, input_div, out_scale = data.save_norm(write=False) # this extracts only the relevant variables
    input_sub = input_sub[None, :]
    input_div = input_div[None, :]
    out_scale = out_scale[None, :]

    lat = grid_info['lat'].values
    lon = grid_info['lon'].values
    lon = grid_info['lon'].values
    lon = ((lon + 180) % 360) - 180
    level = grid_info.lev.values
    lat_bin_mids = data.lat_bin_mids

    ds_nn = xr.open_mfdataset(nn_path)
    ds_sp = xr.open_mfdataset(mmf_path)
    num_hours = ds_nn[var].sizes['time']

    if var == 'T':
        var_name = 'Temperature'
        units = 'K'
        cmap = 'coolwarm'
        vmin = -5
        vmax = 5
    elif var == 'Q':
        var_name = 'Moisture'
        units = 'g/kg'
        cmap = 'coolwarm'
        vmin = -5 # placeholder
        vmax = 5 # placeholder
    elif var == 'CLDLIQ':
        var_name = 'Liquid Cloud'
        units = 'mg/kg'
        cmap = 'coolwarm'
        vmin = -5 # placeholder
        vmax = 5 # placeholder

    arr_nn = ds_nn[var].values
    arr_sp = ds_sp[var].values
    arr_nn_reshaped = np.transpose(arr_nn, (0,2,1))
    arr_sp_reshaped = np.transpose(arr_sp, (0,2,1))
    arr_nn_zonal_mean = data.zonal_bin_weight_3d(arr_nn_reshaped)
    arr_sp_zonal_mean = data.zonal_bin_weight_3d(arr_sp_reshaped)
    arr_nn_zonal_mean_trop = arr_nn_zonal_mean[:,6:12,:].mean(axis = 1)
    arr_sp_zonal_mean_trop = arr_sp_zonal_mean[:,6:12,:].mean(axis = 1)
    arr_nn_zonal_mean_extrop = arr_nn_zonal_mean[:,np.r_[0:6,12:18],:].mean(axis = 1)
    arr_sp_zonal_mean_extrop = arr_sp_zonal_mean[:,np.r_[0:6,12:18],:].mean(axis = 1)
    xr_nn_zonal_mean_trop = xr.DataArray(arr_nn_zonal_mean_trop.T, dims = ['level','time'], \
                                         coords = {'level':level, 'time':np.arange(arr_nn.shape[0])/24.})
    xr_sp_zonal_mean_trop = xr.DataArray(arr_sp_zonal_mean_trop.T, dims = ['level', 'time'], \
                                         coords = {'level':level, 'time':np.arange(arr_sp.shape[0])/24.})
    xr_nn_zonal_mean_extrop = xr.DataArray(arr_nn_zonal_mean_extrop.T, dims = ['level','time'], \
                                         coords = {'level':level, 'time':np.arange(arr_nn.shape[0])/24.})
    xr_sp_zonal_mean_extrop = xr.DataArray(arr_sp_zonal_mean_extrop.T, dims = ['level', 'time'], \
                                         coords = {'level':level, 'time':np.arange(arr_sp.shape[0])/24.})
    bias_trop = xr_nn_zonal_mean_trop - xr_sp_zonal_mean_trop
    bias_extrop = xr_nn_zonal_mean_extrop - xr_sp_zonal_mean_extrop
    
    pressure_weights_sp = make_pressure_weights(grid_info = grid_info, ps = ds_sp['PS'])
    pressure_weights_nn = make_pressure_weights(grid_info = grid_info, ps = ds_nn['PS'])
    
    nn_flatmap = np.sum(ds_nn[var].values*pressure_weights_nn, axis = 1)
    sp_flatmap = np.sum(ds_sp[var].values*pressure_weights_sp, axis = 1)
    
    zonal_min_val = float(arr_nn_zonal_mean.min())
    zonal_max_val = float(arr_sp_zonal_mean.max())
    flatmap_min_val = float(sp_flatmap.min())
    flatmap_max_val = float(sp_flatmap.max())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    ax = axes[0]
    bias_trop.plot(ax=ax, vmin = vmin, vmax = vmax, cmap = cmap)
    ax.invert_yaxis()
    ax.set_xlim(0, max_day)
    ax.set_title(f'(a) Online Bias NN-MMF within 30S-30N: {var_name} {units}',fontsize=14)
    ax.set_xlabel('Days',fontsize=14)
    ax.set_ylabel('Hybrid Pressure (hPa)',fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax = axes[1]
    bias_extrop.plot(ax=ax, vmin = vmin, vmax = vmax, cmap = cmap)
    ax.invert_yaxis()
    ax.set_xlim(0, max_day)
    ax.set_title(f'(b) Online Bias NN-MMF outside 30S-30N: {var_name} {units}',fontsize=14)
    ax.set_xlabel('Days',fontsize=14)
    ax.set_ylabel('Hybrid Pressure (hPa)',fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'first_month_zonal_mean_bias_{var}.png'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get initial online evaluation plots.')
    parser.add_argument('--mmf_path', type=str, required = True, help='Path to the reference MMF simulation.')
    parser.add_argument('--nn_path', type=str, required = True, help='Path to the hybrid simulation.')
    parser.add_argument('--save_path', type=str, required = True, help='Path to save the plots.')
    parser.add_argument('--var', type=str, required = True, help='Variable of interest.')
    parser.add_argument('--max_day', type=int, default = 30, help='Maximum number of days to plot.')
    parser.add_argument('--grid_path', type=str, default='../../grid_info/ClimSim_low-res_grid-info.nc', help='Path to grid info file.')
    parser.add_argument('--input_mean_file', type=str, default = 'input_mean_v2_rh_mc_pervar.nc')
    parser.add_argument('--input_max_file', type=str, default = 'input_max_v2_rh_mc_pervar.nc')
    parser.add_argument('--input_min_file', type=str, default = 'input_min_v2_rh_mc_pervar.nc')
    parser.add_argument('--output_scale_file', type=str, default = 'output_scale_std_lowerthred_v2_rh_mc.nc')
    parser.add_argument('--lbd_qn_file', type=str, default = 'qn_exp_lambda_large.txt')
    args = parser.parse_args()

    main(mmf_path = args.mmf_path, \
         nn_path = args.nn_path, \
         save_path = args.save_path, \
         var = args.var, \
         max_day = args.max_day, \
         grid_path = args.grid_path, \
         input_mean_file = args.input_mean_file, \
         input_max_file = args.input_max_file, \
         input_min_file = args.input_min_file, \
         output_scale_file = args.output_scale_file, \
         lbd_qn_file = args.lbd_qn_file)