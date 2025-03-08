import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    Reshape1,
    Conv1DBlockSqueezeformer,
    TransformerEncoder,
    HeadDense,
    GLUMlp,
    Reshape2,
)
from typing import List
"""
Contains the code for the squeezeformer
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SqueezeformerMetaData(modulus.ModelMetaData):
    name: str = "squeezeformer"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class Squeezeformer(modulus.Module):
    def __init__(self,
                 input_profile_num: int = 9, # number of input profile variables
                 input_scalar_num: int = 17, # number of input scalar variables
                 target_profile_num: int = 5, # number of target profile variables
                 target_scalar_num: int = 8, # number of target scalar variables
                 output_prune: bool = True, # whether or not we prune strato_lev_out levels
                 strato_lev_out: int = 12, # number of levels to set to zero
                 loc_embedding: bool = False, # whether or not to use location embedding
                 embedding_type: str = "positional", # type of location embedding
                 dim: int = 256, # dimension of the model
                 head_dim: int = 1024, # dimension of the head
                ):
        super().__init__(meta = SqueezeformerMetaData())
        self.input_series_num = input_series_num
        self.input_scalar_num = input_scalar_num
        self.target_series_num = target_series_num
        self.target_scalar_num = target_scalar_num
        self.hidden_series_num = hidden_series_num
        self.hidden_scalar_num = hidden_scalar_num
        self.num_hidden_total = self.hidden_series_num + self.hidden_scalar_num
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out

    def forward(self, x):
        # reshape input
        batch_size = x.size(0)
        series_part = x[:, :self.input_series_num*60]
        series_inputs = series_part.reshape(batch_size, self.input_series_num, 60)
        scalar_inputs = x[:, self.input_series_num*60:]

        ###
        # forward pass code here
        ###

        y = torch.cat([series_part, scalar_output], dim = 1)
        # Prune output
        if self.output_prune:
            # Zeyuan says that the .clone() and .clone().zero_() helped bypass a torchscript issue. Reason unclear.
            y = y.clone()
            y[:, 60:60+self.strato_lev_out] = y[:, 60:60+self.strato_lev_out].clone().zero_()
            y[:, 120:120+self.strato_lev_out] = y[:, 120:120+self.strato_lev_out].clone().zero_()
            y[:, 180:180+self.strato_lev_out] = y[:, 180:180+self.strato_lev_out].clone().zero_()
            y[:, 240:240+self.strato_lev_out] = y[:, 240:240+self.strato_lev_out].clone().zero_()
        return y