import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    Conv1DBlockSqueezeformer,
    TransformerEncoder,
    HeadDense,
    GLUMlp,
)
from typing import List
"""
Contains the code for the squeezeformer
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SqueezeformerConfLossMetaData(modulus.ModelMetaData):
    name: str = "squeezeformer_conf_loss"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class SqueezeformerConfLoss(modulus.Module):
    def __init__(self,
                 input_profile_num: int = 9, # number of input profile variables
                 input_scalar_num: int = 17, # number of input scalar variables
                 target_profile_num: int = 5, # number of target profile variables
                 target_scalar_num: int = 8, # number of target scalar variables
                 output_prune: bool = True, # whether or not we prune strato_lev_out levels
                 strato_lev_out: int = 12, # number of levels to set to zero
                 loc_embedding: bool = False, # whether or not to use location embedding
                 embedding_type: str = "positional", # type of location embedding
                 embed_dim: int = 384, # dimension of the model
                 head_dim: int = 2048, # dimension of the head
                 num_heads: int = 4, # number of heads in the transformer
                 num_blocks: int = 12, # number of transformer and squeezeformer blocks
                 conv_filter: int = 15, # filter size for the squeezeformer
                 dilation_rate: int = 1, # dilation rate for the squeezeformer
                 expand_ratio: int = 4, # expand ratio for the squeezeformer
                 activation: str = 'SiLU', # activation function for the squeezeformer
                ):
        super().__init__(meta = SqueezeformerConfLossMetaData())
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.vertical_level_num = 60
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.loc_embedding = loc_embedding
        self.embedding_type = embedding_type
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.conv_filter = conv_filter
        self.dilation_rate = dilation_rate
        self.expand_ratio = expand_ratio
        self.expand_dim = self.embed_dim * self.expand_ratio
        self.channel_size_dwconv = self.expand_dim // 2
        self.activation = activation
        self.dense = nn.Linear(self.input_profile_num + self.input_scalar_num, self.embed_dim, bias = False)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps = 1e-6)
        self.conv_blocks = nn.ModuleList([
            Conv1DBlockSqueezeformer(channel_size = self.embed_dim,
                                     kernel_size = self.conv_filter,
                                     dilation_rate = self.dilation_rate,
                                     expand_ratio = self.expand_ratio,
                                     activation = self.activation) for _ in range(self.num_blocks)
        ])
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim = self.embed_dim,
                               num_heads = self.num_heads,
                               expand_dim = self.expand_dim) for _ in range(self.num_blocks)])
        self.head_dense = HeadDense(self.embed_dim, self.head_dim)
        self.ffn = GLUMlp(expand_dim = self.expand_dim,
                          head_dim = self.head_dim)
        self.final_dense_pred = nn.Linear(in_features = head_dim,
                                          out_features = self.target_profile_num + self.target_scalar_num)
        self.final_dense_conf = nn.Linear(in_features = head_dim,
                                          out_features = self.target_profile_num + self.target_scalar_num)

    def forward(self, x):
        # reshape input
        profile_part = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1,self.input_profile_num,self.vertical_level_num).transpose(1,2)
        scalar_part = x[:,self.input_profile_num*self.vertical_level_num:].unsqueeze(1).repeat(1,self.vertical_level_num,1)
        x = torch.cat([profile_part, scalar_part], dim = -1) # b, 60, 26

        x = self.dense(x)
        x = self.layer_norm(x)
        for conv_block, transformer_encoder in zip(self.conv_blocks, self.transformer_encoders):
            x = conv_block(x)
            x = transformer_encoder(x)
        x = self.head_dense(x)
        x = self.ffn(x)

        outputs_pred = self.final_dense_pred(x)
        outputs_conf = self.final_dense_conf(x)

        profile_part_pred = outputs_pred[:,:,:self.target_profile_num]
        profile_part_pred = profile_part_pred.permute(0,2,1).reshape(-1,self.target_profile_num * self.vertical_level_num) # b,300
        scalar_part_pred = outputs_pred[:,:,self.target_profile_num:]
        scalar_part_pred = torch.mean(scalar_part_pred, dim=1)

        y_pred = torch.concat([profile_part_pred, scalar_part_pred], dim=1) # b, 308

        profile_part_conf = outputs_conf[:,:,:self.target_profile_num]
        profile_part_conf = profile_part_conf.permute(0,2,1).reshape(-1,self.target_profile_num * self.vertical_level_num) # b,300
        scalar_part_conf = outputs_conf[:,:,self.target_profile_num:]
        scalar_part_conf = torch.mean(scalar_part_conf, dim=1)

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