# #import xarray as xr
# from torch.utils.data import Dataset
# import numpy as np
# import torch

#import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import torch

class climsim_dataset(Dataset):
    def __init__(self, 
                 input_paths, 
                 target_paths,
                 norm_arrays,
                 clip = True, 
                 ):
        """
        Args:
            input_paths (str): Path to the .npy file containing the inputs.
            target_paths (str): Path to the .npy file containing the targets.
            norm_arrays (dict): Dictionary containing the normalization arrays.
                norm_arrays = {'input_size': input_size,
                    'output_size': output_size,
                   'mean_y': mean_y,
                   'stds': stds,
                   'mean_col_not': mean_col_not,
                   'std_col_not': std_col_not,
                   'X_total_mean': X_total_mean,
                   'X_total_std': X_total_std,
                   'x_col_mean': x_col_mean,
                   'x_col_std': x_col_std,
                   'x_col_norm_min': x_col_norm_min,
                   'v4_mean_col': v4_mean_col,
                    'v4_std_col': v4_std_col,
                    'v4_mean_total': v4_mean_total,
                    'v4_std_total': v4_std_total,
                   }
        """
        self.inputs = np.load(input_paths)
        self.targets = np.load(target_paths)
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.norm_arrays = norm_arrays
        self.input_size = norm_arrays['input_size']
        self.output_size = norm_arrays['output_size']
        self.col_vars = self.input_size//60
        self.scalar_vars = self.input_size - self.col_vars*60
        self.mean_y = norm_arrays['mean_y']
        self.std_y = norm_arrays['std_y']
        self.stds = norm_arrays['stds']
        self.mean_col_not = norm_arrays['mean_col_not']
        self.std_col_not = norm_arrays['std_col_not']
        self.X_total_mean = norm_arrays['X_total_mean']
        self.X_total_std = norm_arrays['X_total_std']
        self.x_col_mean = norm_arrays['x_col_mean']
        self.x_col_std = norm_arrays['x_col_std']
        self.x_col_norm_min = norm_arrays['x_col_norm_min']
        self.v4_mean_col = norm_arrays['v4_mean_col']
        self.v4_std_col = norm_arrays['v4_std_col']
        self.v4_mean_total = norm_arrays['v4_mean_total']
        self.v4_std_total = norm_arrays['v4_std_total']
        self.clip = clip
        self.mean_y_v5 = np.concatenate([self.mean_y[0:120],self.v4_mean_col[26],self.mean_y[240:368]], axis=0)
        self.stds_v5 = np.concatenate([self.stds[0:120],1./self.v4_std_col[26],self.stds[240:368]], axis=0)
        self.stds_v5[np.isnan(self.stds_v5)] = 0
        self.stds_v5[np.isinf(self.stds_v5)] = 0
        self.mean_col_adv = self.v4_mean_col[6:9].reshape(180)
        self.std_col_adv = self.v4_std_col[6:9].reshape(180)
        self.mean_col_prvphy = np.concatenate([self.mean_y[0:120],self.v4_mean_col[26],self.mean_y[240:300]], axis=0)
        self.std_col_prvphy = np.concatenate([self.std_y[0:120],self.v4_std_col[26],self.std_y[240:300]], axis=0)
        self.std_col_prvphy[60:72] = 0
        self.mean_col_extra = np.concatenate([self.mean_col_adv,self.mean_col_adv,self.mean_col_prvphy,self.mean_col_prvphy], axis=0)
        self.std_col_extra = np.concatenate([self.std_col_adv,self.std_col_adv,self.std_col_prvphy,self.std_col_prvphy], axis=0)

        self.mean_total_adv = self.v4_mean_total[6:9]
        self.std_total_adv = self.v4_std_total[6:9]
        self.mean_total_prvphy = self.v4_mean_total[[12,13,26,16]]
        self.std_total_prvphy = self.v4_std_total[[12,13,26,16]]

        self.mean_total_extra = np.concatenate([self.mean_total_adv,self.mean_total_adv,self.mean_total_prvphy,self.mean_total_prvphy], axis=0)
        self.std_total_extra = np.concatenate([self.std_total_adv,self.std_total_adv,self.std_total_prvphy,self.std_total_prvphy], axis=0)
        self.X_col_qn_norm_min = - self.v4_mean_col[26]/self.v4_std_col[26]
        self.X_col_qn_norm_min[np.isnan(self.X_col_qn_norm_min)] = 0
        self.X_col_qn_norm_min[np.isinf(self.X_col_qn_norm_min)] = 0


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]

        #normalize y
        y_norm = (y - self.mean_y_v5) * self.stds_v5

        #normalize x scalars
        X_col_not = x[self.col_vars*60:self.col_vars*60+17]
        X_col_not = np.delete(X_col_not, -2, axis=0) # exclude cam_in_SNOWHICE
        x_col_not_norm = (X_col_not - self.mean_col_not)/self.std_col_not

        #normalize x columns: t, u,v,3gases
        X_states = np.concatenate([x[:60], x[240:360], x[1200:1380]], axis=0)
        X_total_states_norm = np.reshape(X_states, [6, 60])
        state_idx = [0,4,5,6,7,8]
        X_total_states_norm = (X_total_states_norm - self.X_total_mean[state_idx])/self.X_total_std[state_idx]
        X_total_states_norm = np.reshape(X_total_states_norm, [60*6])

        state_col_mean =  self.x_col_mean.reshape(9,60)[state_idx].reshape(60*6)
        state_col_std =  self.x_col_std.reshape(9,60)[state_idx].reshape(60*6)
        X_col_states_norm = (X_states - state_col_mean)/state_col_std
        X_col_states_norm[np.isnan(X_col_states_norm)] = 0
        X_col_states_norm[np.isinf(X_col_states_norm)] = 0

        #normalize rh:
        X_rh = x[60:120]
        X_col_rh_norm = (X_rh - self.v4_mean_col[1])/self.v4_std_col[1]
        X_total_rh_norm = (X_rh - self.v4_mean_total[1])/self.v4_std_total[1]
        X_col_rh_norm[np.isnan(X_col_rh_norm)] = 0
        X_col_rh_norm[np.isinf(X_col_rh_norm)] = 0

        #normalize qn:
        X_qn = x[120:180]
        X_col_qn_norm = (X_qn - self.v4_mean_col[26])/self.v4_std_col[26]
        X_total_qn_norm = (X_qn - self.v4_mean_total[26])/self.v4_std_total[26]
        X_col_qn_norm[np.isnan(X_col_qn_norm)] = 0
        X_col_qn_norm[np.isinf(X_col_qn_norm)] = 0

        # log qn
        X_col_qn_norm_log = np.log(X_col_qn_norm - self.X_col_qn_norm_min + 1)

        #normalize adv+prevphy tendencies:
        X_extra = x[360:1200]
        X_col_extra_norm = (X_extra - self.mean_col_extra)/self.std_col_extra
        X_total_extra_norm = (X_extra.reshape(14,60) - self.mean_total_extra[:,np.newaxis])/self.std_total_extra[:,np.newaxis]
        X_total_extra_norm = X_total_extra_norm.reshape(14*60)
        X_col_extra_norm[np.isnan(X_col_extra_norm)] = 0
        X_col_extra_norm[np.isinf(X_col_extra_norm)] = 0

        x_col_norm = np.concatenate([X_col_states_norm, X_col_rh_norm, X_col_qn_norm, X_col_extra_norm], axis=0)
        if self.clip:
            cutoff = 30
            square_cutoff = cutoff**0.5
            x_col_norm = np.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
            x_col_norm = np.where(x_col_norm<-cutoff, -np.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)

            X_total_qn_norm = np.where(X_total_qn_norm>cutoff, X_total_qn_norm**0.5+cutoff-square_cutoff, X_total_qn_norm)
            X_total_qn_norm = np.where(X_total_qn_norm<-cutoff, -np.abs(X_total_qn_norm)**0.5-cutoff+square_cutoff, X_total_qn_norm)
        
        x = np.concatenate([X_total_states_norm, X_total_rh_norm, X_total_qn_norm, X_total_extra_norm, x[180:240], x_col_norm, X_col_qn_norm_log, x_col_not_norm], axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)