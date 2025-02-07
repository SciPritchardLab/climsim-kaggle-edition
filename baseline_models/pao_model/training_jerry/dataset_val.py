# #import xarray as xr
# from torch.utils.data import Dataset
# import numpy as np
# import torch

#import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import torch

class dataset_val(Dataset):
    def __init__(self, 
                 input_paths, 
                 target_paths, 
                 output_prune, 
                 strato_lev_out = 12):
        """
        Args:
            input_paths (str): Path to the .npy file containing the inputs.
            target_paths (str): Path to the .npy file containing the targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            qinput_prune (bool): Whether to prune the input data.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
            qc_lbd (np.ndarray): Coefficients for the exponential transformation of qc.
            qi_lbd (np.ndarray): Coefficients for the exponential transformation of qi.
        """
        self.inputs = np.load(input_paths)
        self.targets = np.load(target_paths)
        self.input_paths = input_paths
        self.target_paths = target_paths
        
        self.output_prune = output_prune
        
        self.strato_lev_out = strato_lev_out


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        # x = np.load(self.input_paths,mmap_mode='r')[idx]
        # y = np.load(self.target_paths,mmap_mode='r')[idx]
        
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