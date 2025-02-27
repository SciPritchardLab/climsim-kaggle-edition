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

var = 'T'
max_day = 30
nn_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_allhands/pao_model_debug_ps/run/pao_model_debug_ps.eam.h2.0003-01-*.nc'
sp_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/mmf_runs/mmf_speedeval_gpu/run/mmf_speedeval_gpu.eam.h2.0003-01-*.nc'
save_path = ''

input_mean_file = 'input_mean_v6_pervar.nc'
input_max_file = 'input_max_v6_pervar.nc'
input_min_file = 'input_min_v6_pervar.nc'
output_scale_file = 'output_scale_std_lowerthred_v6.nc'
lbd_qn_file = 'qn_exp_lambda_large.txt'

grid_path = '../../../grid_info/ClimSim_low-res_grid-info.nc'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset('../../../preprocessing/normalizations/inputs/' + input_mean_file)
input_max = xr.open_dataset('../../../preprocessing/normalizations/inputs/' + input_max_file)
input_min = xr.open_dataset('../../../preprocessing/normalizations/inputs/' + input_min_file)
output_scale = xr.open_dataset('../../../preprocessing/normalizations/outputs/' + output_scale_file)
lbd_qn = np.loadtxt('../../../preprocessing/normalizations/inputs/' + lbd_qn_file, delimiter = ',')

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
level = grid_info.lev.values
lat_bin_mids = data.lat_bin_mids

ds_nn = xr.open_mfdataset(nn_path)
ds_sp = xr.open_mfdataset(sp_path)

def make_pressure_weights(grid_info, ps):
    pressure_grid_p1 = np.array(grid_info['P0']*grid_info['hyai'])[np.newaxis,:,np.newaxis]
    pressure_grid_p2 = grid_info['hybi'].values[np.newaxis,:,np.newaxis] * ps.values[:,np.newaxis,:]
    pressure_grid = pressure_grid_p1 + pressure_grid_p2
    dp = pressure_grid[:,1:61,:] - pressure_grid[:,0:60,:]
    pressure_weights = dp/np.sum(dp, axis = 1)[:, np.newaxis, :]
    return pressure_weights

def show_error_first_month(ds_sp, ds_nn, var):
    num_hours = ds_nn[var_name].sizes['time']
    if var == 'T':
        units = 'K'
        cmap = 'RdBu_r'
        vmin = -5
        vmax = 5
    elif var == 'Q':
        cmap = 'Blues'
        var_name = 'Moisture'
        units = 'g/kg'
        vmin = -5 # placeholder
        vmax = 5 # placeholder
    elif var == 'CLDLIQ':
        cmap = 'BrBG'
        var_name = 'Liquid Cloud'
        units = 'mg/kg'
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
    
    nn_flatmap = np.sum(ds_nn[var_name].values*pressure_weights_nn, axis = 1)
    sp_flatmap = np.sum(ds_sp[var_name].values*pressure_weights_sp, axis = 1)
    
    zonal_min_val = float(arr_nn_zonal_mean.min())
    zonal_max_val = float(arr_sp_zonal_mean.max())
    flatmap_min_val = float(sp_flatmap.min())
    flatmap_max_val = float(sp_flatmap.max())

    image_files = []
    if os.path.exists(movie_path):
        shutil.rmtree(movie_path):
    os.makedirs(movie_path)

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
    plt.savefig(os.path.join(, 'first_month_zonal_mean_bias.png'))
    plt.clf()

    for hour in tqdm(range(num_hours)):

        image_file_name = f'{var}/{var}_{str(hour).zfill(5)}.png'
        sp_zonal = xr.DataArray(var_sp_zonal_mean[:,hour,:].T, dims = ['level', 'lat'],
                                        coords={'level': level, 'lat': lat_bin_mids})
        nn_zonal = xr.DataArray(var_nn_zonal_mean[:,hour,:].T, dims = ['level', 'lat'],
                                        coords={'level': level, 'lat': lat_bin_mids})
        
        # Create side-by-side figures and axes with Robinson projections
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)  # Adjust figure size as needed
        gs = fig.add_gridspec(2, 2) 
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        levels = np.linspace(weighted_min_val, weighted_max_val, 20)
        contour_sp = ax1.tricontourf(
            lon, lat, sp_weighted[hour], 
            transform=ccrs.PlateCarree(),  # Data is in lat-lon coordinates
            cmap=cmap,  # Adjust colormap as needed
            levels=levels,  # Number of contour levels
            extend='both',  # Extend beyond data range
            vmin=weighted_min_val,
            vmax=weighted_max_val
        )
        # Add filled contours to the map
        contour_nn = ax2.tricontourf(
            lon, lat, nn_weighted[hour], 
            transform=ccrs.PlateCarree(),  # Data is in lat-lon coordinates
            cmap=cmap,  # Adjust colormap as needed
            levels=levels,  # Number of contour levels
            extend='both',  # Extend beyond data range
            vmin=weighted_min_val,
            vmax=weighted_max_val
        )

        # Add features like coastlines and borders
        ax1.coastlines(linewidth=0.5, color='black')
        ax2.coastlines(linewidth=0.5, color='black')
        # Add a colorbar
        cbar_sp = plt.colorbar(contour_sp, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar_sp.set_label(f'{units}', fontsize = 14)  # Adjust label to match your data
        cbar_sp.locator = ticker.MaxNLocator(nbins=4)

        cbar_nn = plt.colorbar(contour_nn, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar_nn.set_label(f'{units}', fontsize = 14)  # Adjust label to match your data
        cbar_nn.locator = ticker.MaxNLocator(nbins=4)
        # Set title
        ax1.set_title(f'MMF {var_name}: Hour {hour}', fontsize=18)
        ax1.set_global()

        ax2.set_title(f'Hybrid {var_name}: Hour {hour}', fontsize=18)
        ax2.set_global()

        ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        # Now you can plot your data on ax1 and ax2 as needed

        sp_zonal.plot(ax = ax3,
                      vmin = zonal_min_val,
                      vmax = zonal_max_val,
                      cmap = cmap,
                      cbar_kwargs={
                            'orientation': 'vertical',
                            'pad': 0.00,  # Decrease to move the colorbar closer
                            'shrink': 1
                        })
        nn_zonal.plot(ax = ax4,
                      vmin = zonal_min_val,
                      vmax = zonal_max_val,
                      cmap = cmap,
                      cbar_kwargs={
                            'orientation': 'vertical',
                            'pad': 0.00,  # Decrease to move the colorbar closer
                            'shrink': 1
                        })
        
        ax3.set_ylabel('hybrid pressure (hPa)', fontsize = 14)
        ax3.set_xlabel('latitude', fontsize = 14)
        ax4.set_ylabel('hybrid pressure (hPa)', fontsize = 14)
        ax4.set_xlabel('latitude', fontsize = 14)

        
        ax3.tick_params(axis='both', labelsize=14)
        ax4.tick_params(axis='both', labelsize=14)
        
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        
        plt.savefig(image_file_name)
        image_files.append(image_file_name)
        plt.clf()
        plt.close('all')
    video_clip = ImageSequenceClip(sorted(image_files), fps = fps)
    if save_files:
        video_clip.write_videofile(f'{var}.mp4')
    else:
        return video_clip

show_error_first_month(ds_sp, ds_nn, 'T')
