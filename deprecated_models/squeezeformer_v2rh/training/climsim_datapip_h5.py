# #import xarray as xr
# from torch.utils.data import Dataset
# import numpy as np
# import torch

#import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py

class climsim_dataset_h5(Dataset):
    def __init__(self, 
                 parent_path, 
                 norm_arrays,
                 clip = True, 
                 ):
        """
        Args:
            parent_path (str): Path to the .zarr file containing the inputs and targets.
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
        self.parent_path = parent_path
        self.input_paths = glob.glob(f'{parent_path}/**/train_input.h5', recursive=True)
        print('input paths:', self.input_paths)
        if not self.input_paths:
            raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
        self.target_paths = [path.replace('train_input.h5', 'train_target.h5') for path in self.input_paths]

        # Initialize lists to hold the samples count per file
        self.samples_per_file = []
        for input_path in self.input_paths:
            with h5py.File(input_path, 'r') as file:  # Open the file to read the number of samples
                # Assuming dataset is named 'data', adjust if different
                self.samples_per_file.append(file['data'].shape[0])
                
        self.cumulative_samples = np.cumsum([0] + self.samples_per_file)
        self.total_samples = self.cumulative_samples[-1]

        self.input_files = {}
        self.target_files = {}
        for input_path, target_path in zip(self.input_paths, self.target_paths):
            self.input_files[input_path] = h5py.File(input_path, 'r')
            self.target_files[target_path] = h5py.File(target_path, 'r')
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
        return self.total_samples
    
    def _find_file_and_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        local_idx = idx - self.cumulative_samples[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bounds")
        
        file_idx, local_idx = self._find_file_and_index(idx)

        input_file = self.input_files[self.input_paths[file_idx]]
        target_file = self.target_files[self.target_paths[file_idx]]
        x = input_file['data'][local_idx]
        y = target_file['data'][local_idx]

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