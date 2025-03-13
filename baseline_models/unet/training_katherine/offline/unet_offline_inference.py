import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import torch
import os, gc
from climsim_utils.data_utils import *
from tqdm import tqdm


input_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full/scoring_set/scoring_input.npy'
target_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full/scoring_set/scoring_target.npy'
preds_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full/scoring_set/'

output_save_path = '/pscratch/sd/k/kfrields/hugging/scoring/'


grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'
norm_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/'

unet_adam_model_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adam/model.pt'
unet_adamW_model_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adamW/model.pt'
unet_Radam_model_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_Radam/model.pt'

model_paths = {'unet_adam': unet_adam_model_path, 'unet_adamW': unet_adamW_model_path, 'unet_Radam': unet_Radam_model_path}

model_colors = {'unet_adam': 'blue', 'unet_adamW': 'red', 'unet_Radam': 'green'}

model_labels = {'unet_adam': 'unet_adam', 'unet_adamW': 'unet_adamW', 'unet_Radam': 'unet_Radam'}

num_models = len(model_paths)
model_preds = {}
r2_scores = {}
r2_scores_capped = {}

zonal_heating_r2 = {}
zonal_moistening_r2 = {}

rmse_scores = {}
rmse_scores_capped = {}
zonal_heating_rmse = {}
zonal_moistening_rmse = {}


bias_scores = {}
bias_scores_capped = {}
zonal_heating_bias = {}
zonal_moistening_bias = {}

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
                  qinput_log = False)
data.set_to_v2_rh_mc_vars()

input_sub, input_div, out_scale = data.save_norm(write=False) # this extracts only the relevant variables
input_sub = input_sub[None, :]
input_div = input_div[None, :]
out_scale = out_scale[None, :]

lat = grid_info['lat'].values
lon = grid_info['lon'].values
lat_bin_mids = data.lat_bin_mids




def preprocessing_v2_rh_mc(data, input_path, target_path):
    npy_input = np.load(input_path)[:, :]
    npy_target = np.load(target_path)[:, :]

    #de-normalizies the input
    surface_pressure = npy_input[:, data.ps_index] * \
                        (input_max['state_ps'].values - input_min['state_ps'].values) + \
                        input_mean['state_ps'].values
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    npy_input[:,120:180] = 1 - np.exp(-npy_input[:,120:180] * lbd_qn)
    npy_input = (npy_input - input_sub)/input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:,120:120+15] = 0
    npy_input[:,60:120] = np.clip(npy_input[:,60:120], 0, 1.2)
    torch_input = torch.tensor(npy_input).float()
    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)
    return torch_input, reshaped_target, pressures_binned

torch_input, reshaped_target, pressures_binned = preprocessing_v2_rh_mc(data, input_path, target_path)


# model inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model_name in model_paths.keys():
    model = torch.jit.load(model_paths[model_name]).to(device)
    model.eval()
    model_batch_pred_list = []
    batch_size = data.num_latlon
    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size)):
            batch = torch_input[i : i + batch_size, :].to(device)
            model_batch_pred = model(batch) # inference on batch
            model_batch_pred_list.append(model_batch_pred.cpu().numpy() / out_scale)
    model_preds[model_name] = np.stack(model_batch_pred_list, axis = 0) # 0 axis corresponds to time
    np.save(f'{output_save_path}{model_name}_preds.npy', model_preds[model_name])
    
    del model
    del model_batch_pred_list
    gc.collect()


def show_r2(target, preds):
    assert target.shape == preds.shape
    new_shape = (np.prod(target.shape[:-1]), target.shape[-1])
    target_flattened = target.reshape(new_shape)
    preds_flattened = preds.reshape(new_shape)
    r2_scores = np.array([r2_score(target_flattened[:, i], preds_flattened[:, i]) for i in range(308)])
    r2_scores_capped = r2_scores.copy()
    r2_scores_capped[r2_scores_capped < 0] = 0
    return r2_scores, r2_scores_capped



plt.figure(figsize=(10, 6))
for model_name in model_paths.keys():
    r2_scores[model_name], r2_scores_capped[model_name] = show_r2(reshaped_target, model_preds[model_name])
    label_text = f'{model_labels[model_name]}: {np.mean(r2_scores[model_name]):.3g}' 
    plt.plot(np.arange(data.target_feature_len), r2_scores[model_name], color = model_colors[model_name], label=f"{label_text}")
    
plt.legend()
plt.xlabel('Feature Index')
plt.ylabel(r'$R^2$ Score')
plt.title(r'$R^2$ Score Comparison')

feature_indices = [0, 60, 120, 180, 240, 300, 308]
feature_labels = ['dT', 'dQv', 'dQn (liq+ice)', 'dU', 'dV', 'scalars', '']
plt.xticks(ticks=feature_indices, labels=feature_labels)
plt.ylim(0, 1)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig(output_save_path + 'unet_r2_lines.png')
plt.clf()
#plt.show()


def get_coeff(target, pred):
    rss = np.sum((pred - target)**2, axis = 0)
    tss = np.sum((target - np.mean(target, axis = 0)[None,:,:])**2, axis = 0)
    coeff = 1 - rss/tss
    mask = tss == 0
    coeff[mask] = 1.0 * (rss[mask] == 0) 
    return coeff

for model_name in model_paths.keys():
    zonal_heating_r2[model_name] = data.zonal_bin_weight_3d(get_coeff(reshaped_target[:,:,:60],model_preds[model_name][:,:,:60]))[0]
    zonal_moistening_r2[model_name] = data.zonal_bin_weight_3d(get_coeff(reshaped_target[:,:,60:120], model_preds[model_name][:,:,60:120]))[0]
  


fig, ax = plt.subplots(2, num_models, figsize = (num_models*12, 18))
y = np.arange(60)
X, Y = np.meshgrid(np.sin(np.pi*lat_bin_mids/180), y)
Y = (1/100) * np.mean(pressures_binned, axis = 0).T
for i, model_name in enumerate(model_paths.keys()):
    contour_plot_heating = ax[0,i].pcolor(X, Y, zonal_heating_r2[model_name].T, cmap='Blues', vmin = 0, vmax = 1)
    ax[0,i].contour(X, Y, zonal_heating_r2[model_name].T, [0.7], colors='orange', linewidths=[4])
    ax[0,i].contour(X, Y, zonal_heating_r2[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
    ax[0,i].set_title(f'{model_labels[model_name]} (heating)', fontsize = 20)
    ax[0,i].set_xticks([])
    contour_plot = ax[1,i].pcolor(X, Y, zonal_moistening_r2[model_name].T, cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
    ax[1,i].contour(X, Y, zonal_moistening_r2[model_name].T, [0.7], colors='orange', linewidths=[4])
    ax[1,i].contour(X, Y, zonal_moistening_r2[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
    ax[1,i].set_title(f'{model_labels[model_name]} (moistening)', fontsize = 20)
    ax[1,i].xaxis.set_ticks([np.sin(-50/180*np.pi), 0, np.sin(50/180*np.pi)])
    ax[1,i].xaxis.set_ticklabels(['50$^\circ$S', '0$^\circ$', '50$^\circ$N'], fontsize = 16)
    ax[1,i].xaxis.set_tick_params(width = 2)
    if i != 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
ax[0,0].set_ylabel("Pressure [hPa]", fontsize = 22)
ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
ax[0,0].yaxis.set_tick_params(labelsize = 14)
ax[1,0].yaxis.set_tick_params(labelsize = 14)
ax[0,0].yaxis.set_ticks([1000,800,600,400,200,0])
ax[1,0].yaxis.set_ticks([1000,800,600,400,200,0])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
cb = fig.colorbar(contour_plot, cax=cbar_ax)
cb.set_label("Skill Score "+r'$\left(\mathrm{R^{2}}\right)$',labelpad=50.1, fontsize = 20)
plt.suptitle("Baseline Models Skill for Vertically Resolved Tendencies", y = 0.97, fontsize = 22)
plt.subplots_adjust(hspace=0.1)
plt.savefig(output_save_path + 'unet_press_lat_r2_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
plt.clf()




def show_rmse(target, preds):
    assert target.shape == preds.shape
    new_shape = (np.prod(target.shape[:-1]), target.shape[-1])
    target_flattened = target.reshape(new_shape)
    preds_flattened = preds.reshape(new_shape)
    #print(preds_flattened.shape)
    rmse_scores = np.array([np.sqrt(mean_squared_error(target_flattened[:, i], preds_flattened[:, i])) for i in range(308)])
    #rint(rmse_scores.shape)
    rmse_scores_capped = rmse_scores.copy()
    rmse_scores_capped[rmse_scores_capped < 0] = 0
    return rmse_scores, rmse_scores_capped


feature_indices = [0, 60, 120, 180, 240, 300, 308]

fig, ax = plt.subplots(5, figsize = (10,15))
for model_name in model_paths.keys():
    rmse_scores[model_name], rmse_scores_capped[model_name] = show_rmse(reshaped_target, model_preds[model_name])
    label_text = f'{model_labels[model_name]}: {np.mean(rmse_scores[model_name]):.3g}' 

    for feature_i in range(5):
        starting_index = feature_indices[feature_i]
        ending_index = feature_indices[feature_i + 1]
        xaxis = np.arange(ending_index - starting_index)
        ax[feature_i].plot(xaxis, rmse_scores[model_name][starting_index: ending_index], color = model_colors[model_name], label=f"{label_text}")

    
feature_labels = ['dT', 'dQv', 'dQn (liq+ice)', 'dU', 'dV']
feature_units = ['(K/s)', '(kg/kg/s)', '(kg/kg/s)', '(m/$s^2$)', '(m/$s^2$)']
for feature_i in range(5):
    ax[feature_i].set_title(f'{feature_labels[feature_i]}')
    ax[feature_i].set_ylabel(f'RMSE {feature_units[feature_i]}')
    ax[feature_i].legend()
    
fig.tight_layout()
plt.savefig(output_save_path + 'unet_rmse_lines.png')
#plt.show()
plt.clf()



fig, ax = plt.subplots(4,2, figsize = (8,15))
scalar_start_index = 300
scalar_labels = ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC', 'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL', 'cam_out_SOLSD','cam_out_SOLLD']
scalar_units = ['W/$m^2$', 'W/$m^2$', 'm/s', 'm/s', 'w/$m^2$', 'w/$m^2$', 'w/$m^2$', 'w/$m^2$']
models =  model_paths.keys()

for scalar_i, scalar_name in enumerate(scalar_labels):
    scalar_rmse = []
    
    for model_i, model_name in enumerate(models):
        scalar_rmse.append(rmse_scores[model_name][scalar_start_index + scalar_i])
    #bar_colors = ['tab:blue', 'tab:red', 'tab:green']
    bar_colors = model_colors.values()
    ax[scalar_i//2][scalar_i%2].bar(models, scalar_rmse, color = bar_colors)
    ax[scalar_i//2][scalar_i%2].set_title(scalar_name)
    ax[scalar_i//2][scalar_i%2].set_ylabel(f'RMSE ({scalar_units[scalar_i]})')
    ax[scalar_i//2][scalar_i%2].tick_params(axis='x', labelrotation=90)
    
fig.tight_layout()
plt.savefig(output_save_path + 'unet_rmse_scalar_bars.png')
#plt.show()
plt.clf()





def get_rmse_coeff(target, pred):
    n = target.shape[0]
    num = np.sum((target - np.mean(target, axis = 0)[None,:,:])**2, axis = 0)
    coeff = np.sqrt(np.divide(num,n))
    return coeff


for model_name in model_paths.keys():
    zonal_heating_rmse[model_name] = data.zonal_bin_weight_3d(get_rmse_coeff(reshaped_target[:,:,:60],model_preds[model_name][:,:,:60]))[0]
    zonal_moistening_rmse[model_name] = data.zonal_bin_weight_3d(get_rmse_coeff(reshaped_target[:,:,60:120], model_preds[model_name][:,:,60:120]))[0]
  
fig, ax = plt.subplots(2, num_models, figsize = (num_models*12, 18))
y = np.arange(60)
X, Y = np.meshgrid(np.sin(np.pi*lat_bin_mids/180), y)
Y = (1/100) * np.mean(pressures_binned, axis = 0).T
for i, model_name in enumerate(model_paths.keys()):
    contour_plot_heating = ax[0,i].pcolor(X, Y, zonal_heating_rmse[model_name].T, cmap='Reds')
    #ax[0,i].contour(X, Y, zonal_heating_rmse[model_name].T, [0.7], colors='orange', linewidths=[4])
    #ax[0,i].contour(X, Y, zonal_heating_rmse[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
    ax[0,i].set_title(f'{model_labels[model_name]} (heating)', fontsize = 20)
    ax[0,i].set_xticks([])
    contour_plot_cooling = ax[1,i].pcolor(X, Y, zonal_moistening_rmse[model_name].T, cmap='Blues') # pcolormesh
    #ax[1,i].contour(X, Y, zonal_moistening_rmse[model_name].T, [0.7], colors='orange', linewidths=[4])
    #ax[1,i].contour(X, Y, zonal_moistening_rmse[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
    ax[1,i].set_title(f'{model_labels[model_name]} (moistening)', fontsize = 20)
    ax[1,i].xaxis.set_ticks([np.sin(-50/180*np.pi), 0, np.sin(50/180*np.pi)])
    ax[1,i].xaxis.set_ticklabels(['50$^\circ$S', '0$^\circ$', '50$^\circ$N'], fontsize = 16)
    ax[1,i].xaxis.set_tick_params(width = 2)
    if i != 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
ax[0,0].set_ylabel("Pressure [hPa]", fontsize = 22)
ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
ax[0,0].yaxis.set_tick_params(labelsize = 14)
ax[1,0].yaxis.set_tick_params(labelsize = 14)
ax[0,0].yaxis.set_ticks([1000,800,600,400,200,0])
ax[1,0].yaxis.set_ticks([1000,800,600,400,200,0])
fig.subplots_adjust(right=0.8)
#acbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
#cbh = fig.colorbar(contour_plot_heating, cax=cbar_ax)
cbh = fig.colorbar(contour_plot_heating)
cbc = fig.colorbar(contour_plot_cooling)
cbh.set_label("Skill Score "+r'$\left(\mathrm{RMSE}\right)$ (K/s)',labelpad=50.1, fontsize = 20)
cbc.set_label("Skill Score "+r'$\left(\mathrm{RMSE}\right)$ (kg/kg/s)',labelpad=50.1, fontsize = 20)
plt.suptitle("Baseline Models Skill for Vertically Resolved Tendencies", y = 0.97, fontsize = 22)
plt.subplots_adjust(hspace=0.1)
plt.savefig(output_save_path + 'unet_press_lat_rmse_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)



def show_bias(target, preds):
    assert target.shape == preds.shape
    new_shape = (np.prod(target.shape[:-1]), target.shape[-1])
    target_flattened = target.reshape(new_shape)
    preds_flattened = preds.reshape(new_shape)
    #print(preds_flattened.shape)
    #bias_scores = np.array([(np.mean(preds[:,:, i], axis = 0)-np.mean(target[:,:, i], axis = 0)) for i in range(308)])
    bias_scores = np.subtract(np.mean(preds_flattened, axis = 0), np.mean(target_flattened, axis = 0))
    #print(bias_scores.shape)
    bias_scores_capped = bias_scores.copy()
    bias_scores_capped[bias_scores_capped < 0] = 0
    return bias_scores, bias_scores_capped

feature_indices = [0, 60, 120, 180, 240, 300, 308]

fig, ax = plt.subplots(5, figsize = (10,15))
for model_name in model_paths.keys():
    bias_scores[model_name], bias_scores_capped[model_name] = show_bias(reshaped_target, model_preds[model_name])
    label_text = f'{model_labels[model_name]}: {np.mean(bias_scores[model_name]):.3g}' 

    for feature_i in range(5):
        starting_index = feature_indices[feature_i]
        ending_index = feature_indices[feature_i + 1]
        xaxis = np.arange(ending_index - starting_index)
        ax[feature_i].plot(xaxis, bias_scores[model_name][starting_index: ending_index], color = model_colors[model_name], label=f"{label_text}")

feature_labels = ['dT', 'dQv', 'dQn (liq+ice)', 'dU', 'dV']
feature_units = ['(K/s)', '(kg/kg/s)', '(kg/kg/s)', '(m/$s^2$)', '(m/$s^2$)']

for feature_i in range(5):
    ax[feature_i].set_title(f'{feature_labels[feature_i]}')
    ax[feature_i].set_ylabel(f'BIAS {feature_units[feature_i]}')
    ax[feature_i].legend()

fig.tight_layout()
plt.savefig(output_save_path + 'unet_bias_lines.png')
#plt.show()
plt.clf()


fig, ax = plt.subplots(4,2, figsize = (12,15))
scalar_start_index = 300
scalar_labels = ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC', 'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL', 'cam_out_SOLSD','cam_out_SOLLD']
scalar_units = ['W/$m^2$', 'W/$m^2$', 'm/s', 'm/s', 'w/$m^2$', 'w/$m^2$', 'w/$m^2$', 'w/$m^2$']

models =  model_paths.keys()

for scalar_i, scalar_name in enumerate(scalar_labels):
    scalar_bias = []
    
    for model_i, model_name in enumerate(models):
        scalar_bias.append(bias_scores[model_name][scalar_start_index + scalar_i])
    #bar_colors = ['tab:blue', 'tab:red', 'tab:green']
    bar_colors = model_colors.values()
    bar_labels = bar_labels = model_colors.keys()
    ax[scalar_i//2][scalar_i%2].bar(models, scalar_bias, width = .5,label = bar_labels, color = bar_colors)
    ax[scalar_i//2][scalar_i%2].set_title(scalar_name)
    ax[scalar_i//2][scalar_i%2].set_xticklabels([])
    ax[scalar_i//2][scalar_i%2].set_ylabel(f'Bias ({scalar_units[scalar_i]})')
    ax[scalar_i//2][scalar_i%2].spines['bottom'].set_position('center')
    ax[scalar_i//2][scalar_i%2].spines['bottom'].set_position(('data', 0))

    ax[scalar_i//2][scalar_i%2].legend()
    
fig.tight_layout()
plt.savefig(output_save_path + 'unet_bias_scalar_bars.png')
#plt.show()
plt.clf()


def get_bias_coeff(target, pred):
    coeff = np.subtract(np.mean(pred, axis = 0), np.mean(target, axis = 0))
    #num = np.sum((target - np.mean(target, axis = 0)[None,:,:])**2, axis = 0)
    #coeff = np.sqrt(np.divide(num,n))
    #print(coeff.shape)
    return coeff


for model_name in model_paths.keys():
    zonal_heating_bias[model_name] = data.zonal_bin_weight_3d(get_bias_coeff(reshaped_target[:,:,:60],model_preds[model_name][:,:,:60]))[0]
    zonal_moistening_bias[model_name] = data.zonal_bin_weight_3d(get_bias_coeff(reshaped_target[:,:,60:120], model_preds[model_name][:,:,60:120]))[0]
  

fig, ax = plt.subplots(2, num_models, figsize = (num_models*12, 18))
y = np.arange(60)
X, Y = np.meshgrid(np.sin(np.pi*lat_bin_mids/180), y)
Y = (1/100) * np.mean(pressures_binned, axis = 0).T
for i, model_name in enumerate(model_paths.keys()):
    contour_plot_heating = ax[0,i].pcolor(X, Y, zonal_heating_bias[model_name].T, cmap='RdBu')
    #ax[0,i].contour(X, Y, zonal_heating_rmse[model_name].T, [0.7], colors='orange', linewidths=[4])
    #ax[0,i].contour(X, Y, zonal_heating_rmse[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
    ax[0,i].set_title(f'{model_labels[model_name]} (heating)', fontsize = 20)
    ax[0,i].set_xticks([])
    contour_plot_cooling = ax[1,i].pcolor(X, Y, zonal_moistening_bias[model_name].T, cmap='PuOr') # pcolormesh
    #ax[1,i].contour(X, Y, zonal_moistening_rmse[model_name].T, [0.7], colors='orange', linewidths=[4])
    #ax[1,i].contour(X, Y, zonal_moistening_rmse[model_name].T, [0.9], colors='yellow', linewidths=[4])
    ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
    ax[1,i].set_title(f'{model_labels[model_name]} (moistening)', fontsize = 20)
    ax[1,i].xaxis.set_ticks([np.sin(-50/180*np.pi), 0, np.sin(50/180*np.pi)])
    ax[1,i].xaxis.set_ticklabels(['50$^\circ$S', '0$^\circ$', '50$^\circ$N'], fontsize = 16)
    ax[1,i].xaxis.set_tick_params(width = 2)
    if i != 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
ax[0,0].set_ylabel("Pressure [hPa]", fontsize = 22)
ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
ax[0,0].yaxis.set_tick_params(labelsize = 14)
ax[1,0].yaxis.set_tick_params(labelsize = 14)
ax[0,0].yaxis.set_ticks([1000,800,600,400,200,0])
ax[1,0].yaxis.set_ticks([1000,800,600,400,200,0])
fig.subplots_adjust(right=0.8)
#acbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
#cbh = fig.colorbar(contour_plot_heating, cax=cbar_ax)
cbh = fig.colorbar(contour_plot_heating)
cbc = fig.colorbar(contour_plot_cooling)
cbh.set_label("Skill Score "+r'$\left(\mathrm{Bias}\right)$ (K/s)',labelpad=50.1, fontsize = 20)
cbc.set_label("Skill Score "+r'$\left(\mathrm{Bias}\right)$ (kg/kg/s)',labelpad=50.1, fontsize = 20)
plt.suptitle("Baseline Models Skill for Vertically Resolved Tendencies", y = 0.97, fontsize = 22)
plt.subplots_adjust(hspace=0.1)
plt.savefig(output_save_path + 'unet_press_lat_bias_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
plt.clf()





