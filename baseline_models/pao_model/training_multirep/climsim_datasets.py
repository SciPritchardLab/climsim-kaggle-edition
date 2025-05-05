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

class TrainingDataset(Dataset):
    def __init__(self, 
                 parent_path, 
                 input_sub_per_lev, 
                 input_div_per_lev,
                 input_sub_per_col,
                 input_div_per_col,
                 input_min_norm_per_lev, 
                 out_scale,
                 output_prune,
                 strato_lev,
                 strato_lev_out = 12,):
        """
        Args:
            parent_path (str): Path to the .zarr file containing the inputs and targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
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

        # for input_path, target_path in zip(self.input_paths, self.target_paths):
        #     # Lazily open zarr files and keep the reference
        #     self.input_zarrs[input_path] = zarr.open(input_path, mode='r')
        #     self.target_zarrs[target_path] = zarr.open(target_path, mode='r')
        
        self.input_sub_per_lev = input_sub_per_lev
        self.input_div_per_lev = input_div_per_lev
        self.input_sub_per_col = input_sub_per_col
        self.input_div_per_col = input_div_per_col
        self.input_min_norm_per_lev = input_min_norm_per_lev
        self.out_scale = out_scale
        self.output_prune = output_prune
        self.strato_lev = strato_lev
        self.strato_lev_out = strato_lev_out

    def __len__(self):
        return self.total_samples
    
    def _find_file_and_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        local_idx = idx - self.cumulative_samples[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bounds")
        # Find which file the index falls into
        # file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        # local_idx = idx - self.cumulative_samples[file_idx]

        # x = zarr.open(self.input_paths[file_idx], mode='r')[local_idx]
        # y = zarr.open(self.target_paths[file_idx], mode='r')[local_idx]
        file_idx, local_idx = self._find_file_and_index(idx)


        # x = self.input_zarrs[self.input_paths[file_idx]][local_idx]
        # y = self.target_zarrs[self.target_paths[file_idx]][local_idx]
        # Open the HDF5 files and read the data for the given index
        input_file = self.input_files[self.input_paths[file_idx]]
        target_file = self.target_files[self.target_paths[file_idx]]
        x = input_file['data'][local_idx]
        y = target_file['data'][local_idx]

        # with h5py.File(self.input_paths[file_idx], 'r') as input_file:
        #     x = input_file['data'][local_idx]  # Adjust 'data' if your dataset has a different name
        
        # with h5py.File(self.target_paths[file_idx], 'r') as target_file:
        #     y = target_file['data'][local_idx]  # Adjust 'data' if your dataset has a different name

        # x = np.load(self.input_paths,mmap_mode='r')[idx]
        # y = np.load(self.target_paths,mmap_mode='r')[idx]
        # Avoid division by zero in input_div and set corresponding x to 0
        # input_div_nonzero = self.input_div != 0
        # x = np.where(input_div_nonzero, (x - self.input_sub) / self.input_div, 0)

        x1 = (x - self.input_sub_per_lev) / self.input_div_per_lev
        x1[np.isnan(x1)] = 0
        x1[np.isinf(x1)] = 0
        x2 = (np.concatenate([x[:180], x[240:540]]) - self.input_sub_per_col) / self.input_div_per_col
        x2[np.isnan(x2)] = 0
        x2[np.isinf(x2)] = 0
        x_col_norm = np.concatenate([x1[:180], x1[240:540]])
        x3 = np.where(x_col_norm >= self.input_min_norm_per_lev, \
                      np.log((x_col_norm - self.input_min_norm_per_lev) + 1), \
                      -np.log((self.input_min_norm_per_lev - x_col_norm) + 1))
        x = np.concatenate([x1[:540], x2, x3, x1[540:]])
        #make all inf and nan values 0
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        y = y * self.out_scale

        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ValidationDataset(Dataset):
    def __init__(self, 
                 val_input_path, 
                 val_target_path, 
                 input_sub_per_lev, 
                 input_div_per_lev,
                 input_sub_per_col,
                 input_div_per_col,
                 input_min_norm_per_lev, 
                 out_scale,
                 output_prune, 
                 strato_lev,
                 strato_lev_out = 12,):
        """
        Args:
            val_input_path (str): Path to the .npy file containing the inputs.
            val_target_path (str): Path to the .npy file containing the targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
        """
        self.val_input = np.load(val_input_path)
        self.val_target = np.load(val_target_path)
        self.input_sub_per_lev = input_sub_per_lev
        self.input_div_per_lev = input_div_per_lev
        self.input_sub_per_col = input_sub_per_col
        self.input_div_per_col = input_div_per_col
        self.input_min_norm_per_lev = input_min_norm_per_lev
        self.out_scale = out_scale
        self.output_prune = output_prune
        self.strato_lev = strato_lev
        self.strato_lev_out = strato_lev_out
        assert len(self.val_input) == len(self.val_target)

    def __len__(self):
        return len(self.val_input)

    def __getitem__(self, idx):
        x = self.val_input[idx]
        y = self.val_target[idx]
        # Avoid division by zero in input_div and set corresponding x to 0
        # input_div_nonzero = self.input_div != 0
        # x = np.where(input_div_nonzero, (x - self.input_sub) / self.input_div, 0)
        x1 = (x - self.input_sub_per_lev) / self.input_div_per_lev
        x1[np.isnan(x1)] = 0
        x1[np.isinf(x1)] = 0
        x2 = (np.concatenate([x[:180], x[240:540]]) - self.input_sub_per_col) / self.input_div_per_col
        x2[np.isnan(x2)] = 0
        x2[np.isinf(x2)] = 0
        x_col_norm = np.concatenate([x1[:180], x1[240:540]])
        x3 = np.where(x_col_norm >= self.input_min_norm_per_lev, \
                      np.log((x_col_norm - self.input_min_norm_per_lev) + 1), \
                      -np.log((self.input_min_norm_per_lev - x_col_norm) + 1))
        x = np.concatenate([x1[:540], x2, x3, x1[540:]])
        #make all inf and nan values 0
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0

        y = y * self.out_scale

        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)