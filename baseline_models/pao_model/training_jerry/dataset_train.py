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

class dataset_train(Dataset):
    def __init__(self, 
                 parent_path,
                 output_prune,
                 strato_lev_out=12
                 ):
        """
        Args:
            parent_path (str): Path to the .zarr file containing the inputs and targets.
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
        self.output_prune = output_prune
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

        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0
            # y[300:300+self.strato_lev_out] = 0
        
        # Convert numpy arrays to torch tensors with float32 dtype
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        return x, y
    