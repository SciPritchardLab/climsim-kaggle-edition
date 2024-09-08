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
        self.stds = norm_arrays['stds']
        self.mean_col_not = norm_arrays['mean_col_not']
        self.std_col_not = norm_arrays['std_col_not']
        self.X_total_mean = norm_arrays['X_total_mean']
        self.X_total_std = norm_arrays['X_total_std']
        self.x_col_mean = norm_arrays['x_col_mean']
        self.x_col_std = norm_arrays['x_col_std']
        self.x_col_norm_min = norm_arrays['x_col_norm_min']
        self.clip = clip


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]

        y_norm = (y - self.mean_y) * self.stds
        X_col_not = x[self.col_vars*60:]
        X_col_not = np.delete(X_col_not, -2, axis=0) # exclude cam_in_SNOWHICE
        X_col = x[:self.col_vars*60]
        x_col_not_norm = (X_col_not - self.mean_col_not)/self.std_col_not
        x_total_norm = np.reshape(X_col, [9, 60])
        x_total_norm = (x_total_norm - self.X_total_mean)/self.X_total_std
        x_total_norm = np.reshape(x_total_norm, [60*9])

        x_col_norm = (X_col - self.x_col_mean)/self.x_col_std
        x_col_norm[np.isnan(x_col_norm)] = 0
        x_col_norm[np.isinf(x_col_norm)] = 0
        x_col_norm_log = np.log(x_col_norm[120:240] - self.x_col_norm_min[120:240]+ 1)

        if self.clip:
            cutoff = 30
            square_cutoff = cutoff**0.5
            x_col_norm = np.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
            x_col_norm = np.where(x_col_norm<-cutoff, -np.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)
        
        # x = np.concatenate([x_total_norm, x_col_norm, x_col_norm_log, x_col_not_norm], axis=0)
        x = np.concatenate([x_total_norm, x_col_norm, x_col_norm_log, x_col_not_norm], axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)