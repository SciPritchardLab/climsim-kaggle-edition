import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    DownsampleLayer,
    PointWiseLinear,
    Block,
    DoubleBlock,
)
from typing import List
"""
Contains the code for the ConvNeXt model
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ConvNeXtConfLossMetaData(modulus.ModelMetaData):
    name: str = "convnext_conf_loss"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class ConvNeXtConfLoss(modulus.Module):
    def __init__(self,
                 input_profile_num: int = 9, # number of input profile variables
                 input_scalar_num: int = 17, # number of input scalar variables
                 target_profile_num: int = 5, # number of target profile variables
                 target_scalar_num: int = 8, # number of target scalar variables
                 output_prune: bool = True, # whether or not we prune strato_lev_out levels
                 strato_lev_out: int = 12, # number of levels to set to zero
                 loc_embedding: bool = False, # whether or not to use location embedding
                 embedding_type: str = "positional", # type of location embedding
                 depths: List[int] = [3, 3, 9, 3], # number of blocks in each stage
                 kernel_sizes: List[int] = [4, 2, 2, 2], # kernel size for each stage
                 dims: List[int] = [96, 192, 384, 768], # Feature dimension at each stage
                 layer_scale_init_value: float = 1e-6, # initial value for layer scale
                 block_type: str = 'block', # block type
                 block_kernel_size: int = 7, # block kernel size
                ):
        super().__init__(meta = ConvNeXtConfLossMetaData())
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.vertical_level_num = 60
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.loc_embedding = loc_embedding
        self.embedding_type = embedding_type
        self.depths = depths
        self.kernel_sizes = kernel_sizes
        self.dims = dims
        self.layer_scale_init_value = layer_scale_init_value
        self.block_type = block_type
        self.block_kernel_size = block_kernel_size
        self.num_input_channels = self.input_profile_num + self.input_scalar_num
        self.downsample_layers = (
            nn.ModuleList()
        )
        stem = nn.Sequential(
            nn.Conv1d(
                in_channels = self.num_input_channels,
                out_channels = self.dims[0],
                kernel_size = kernel_sizes[0],
                stride = 1,
                padding = 'same',
            ),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            self.downsample_layers.append(DownsampleLayer(in_channels = dims[i], \
                                                          out_channels = dims[i + 1], \
                                                          kernel_size = kernel_sizes[i + 1]))

        self.stages = (
            nn.ModuleList()
        )
        cur = 0
        for i in range(len(depths)):
            if block_type == 'block':
                blocks = [
                    Block(
                        dim = dims[i],
                        drop_path = 0.0,
                        layer_scale_init_value = layer_scale_init_value,
                        pointwise = PointWiseLinear,
                        kernel_size = block_kernel_size,
                    )
                    for j in range(depths[i])
                ]
            elif block_type == 'doubleblock':
                blocks = [
                    DoubleBlock(
                        dim = dims[i],
                        drop_path = 0.0,
                        layer_scale_init_value = layer_scale_init_value,
                        pointwise = PointWiseLinear,
                        kernel_size = block_kernel_size,
                    )
                    for j in range(depths[i])
                ]
            else:
                raise ValueError(f"block_type: {block_type} is not supported.")
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]

            self.final_conv_layer_pred = nn.Conv1d(
                in_channels = dims[-1],
                out_channels = target_profile_num + target_scalar_num,
                kernel_size = 3,
                stride = 1,
                padding = 'same',
            )
            self.fc_scalar_pred = nn.Linear(
                in_features = self.target_scalar_num * self.vertical_level_num,
                out_features = self.target_scalar_num,
            )

            self.final_conv_layer_conf = nn.Conv1d(
                in_channels = dims[-1],
                out_channels = target_profile_num + target_scalar_num,
                kernel_size = 3,
                stride = 1,
                padding = 'same',
            )
            self.fc_scalar_conf = nn.Linear(
                in_features = self.target_scalar_num * self.vertical_level_num,
                out_features = self.target_scalar_num,
            )

    def forward(self, x):
        x_profile = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1, self.input_profile_num, self.vertical_level_num)
        x_scalar = x[:,self.input_profile_num*self.vertical_level_num:].unsqueeze(2).repeat(1, 1, self.vertical_level_num)
        x = torch.cat((x_profile, x_scalar), dim=1) # b, 26, 60

        for downsample_layer, stage in zip(self.downsample_layers, self.stages):
            x = downsample_layer(x)
            x = stage(x)
        
        outputs_pred = self.final_conv_layer_pred(x)
        outputs_conf = self.final_conv_layer_conf(x)

        profile_part_pred = outputs_pred[:,:self.target_profile_num,:]
        profile_part_pred = profile_part_pred.reshape(-1,self.target_profile_num * self.vertical_level_num) # b,300
        scalar_part_pred = outputs_pred[:,self.target_profile_num:,:]
        scalar_part_pred = self.fc_scalar_pred(scalar_part_pred.reshape(-1,self.target_scalar_num * self.vertical_level_num)) # b, 8

        y_pred = torch.concat([profile_part_pred, scalar_part_pred], dim=1) # b, 308

        profile_part_conf = outputs_conf[:,:self.target_profile_num,:]
        profile_part_conf = profile_part_conf.reshape(-1,self.target_profile_num * self.vertical_level_num) # b,300
        scalar_part_conf = outputs_conf[:,self.target_profile_num:,:]
        scalar_part_conf = self.fc_scalar_conf(scalar_part_conf.reshape(-1,self.target_scalar_num * self.vertical_level_num)) # b, 8

        y_conf = torch.concat([profile_part_conf, scalar_part_conf], dim=1) # b, 308

        # Prune output
        if self.output_prune:
            # Zeyuan says that the .clone() and .clone().zero_() helped bypass a torchscript issue. Reason unclear.
            y_pred = y_pred.clone()
            y_pred[:, 60:60+self.strato_lev_out] = y_pred[:, 60:60+self.strato_lev_out].clone().zero_()
            y_pred[:, 120:120+self.strato_lev_out] = y_pred[:, 120:120+self.strato_lev_out].clone().zero_()
            y_pred[:, 180:180+self.strato_lev_out] = y_pred[:, 180:180+self.strato_lev_out].clone().zero_()
            y_pred[:, 240:240+self.strato_lev_out] = y_pred[:, 240:240+self.strato_lev_out].clone().zero_()

            y_conf = y_conf.clone()
            y_conf[:, 60:60+self.strato_lev_out] = y_conf[:, 60:60+self.strato_lev_out].clone().zero_()
            y_conf[:, 120:120+self.strato_lev_out] = y_conf[:, 120:120+self.strato_lev_out].clone().zero_()
            y_conf[:, 180:180+self.strato_lev_out] = y_conf[:, 180:180+self.strato_lev_out].clone().zero_()
            y_conf[:, 240:240+self.strato_lev_out] = y_conf[:, 240:240+self.strato_lev_out].clone().zero_()
        return y_pred, y_conf