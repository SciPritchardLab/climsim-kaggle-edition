# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from models.submodules.layer_modules import DropPath, ScaleBiasLayer
# from models.submodules.masked_batchnorm import MaskedBatchNorm1d
# from models.submodules.masked_conv import MaskedConv1d

# def get_act_fn(activation):
#     if activation == 'swish':
#         return nn.SiLU()
#     elif activation == 'silu':
#         return nn.SiLU()
#     elif activation == 'gelu':
#         return nn.GELU()
#     elif activation == 'relu':
#         return nn.ReLU()
#     elif activation == 'mish':
#         return nn.Mish()
#     elif activation == 'prelu':
#         return nn.PReLU()
#     elif activation == 'elu':
#         return nn.ELU()
#     else:
#         raise NotImplmentedError

# class GLU(nn.Module):
#     def __init__(self, dim: int = -1, activation: str = 'swish') -> None:
#         super(GLU, self).__init__()
#         self.dim = dim
#         self.activation = get_act_fn(activation)

#     def forward(self, inputs: Tensor) -> Tensor:
#         outputs, gate = inputs.chunk(2, dim=self.dim)
#         return outputs * self.activation(gate)

# class GLUMlp(nn.Module):
#     def __init__(
#         self,
#         dim: int = 512,
#         expand: int = 4,
#         bias : bool = True,
#         activation: str = 'swish'
#     ) -> None:
#         super(GLUMlp, self).__init__()

#         self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
#         self.glu = GLU(dim=-1, activation=activation)
#         self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
#         # self.do2 = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x = self.ffn1(x)
#         x = self.glu(x)
#         x = self.ffn2(x)
#         return x
    
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        self.activation = nn.SiLU()  # Equivalent to 'swish' in TensorFlow/Keras
    def forward(self, x):
        x, gate = torch.chunk(x, 2, dim=-1)
        x = x * self.activation(gate)
        return x
    
class GLUMlp(nn.Module):
    def __init__(self, expand_dim, head_dim):
        super(GLUMlp, self).__init__()
        self.dense_1 = nn.Linear(head_dim, expand_dim)
        self.glu_1 = GLU()
        self.dense_2 = nn.Linear(expand_dim // 2, head_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.glu_1(x)
        x = self.dense_2(x)
        return x
    
class ScaleBias(nn.Module):
    def __init__(self, num_features):
        super(ScaleBias, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.bias
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, expand_dim):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = GLUMlp(expand_dim, embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.scale_bias_1 = ScaleBias(embed_dim)
        self.scale_bias_2 = ScaleBias(embed_dim)

    def forward(self, x):
        residual = x
        x, _ = self.att(x, x, x)
        x = self.scale_bias_1(x)
        x = self.layer_norm_1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x
    
class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding='same', bias=False)

    def forward(self, inputs):
        # Global average pooling
        x = torch.mean(inputs, dim=-1, keepdim=True)  # Shape: (batch_size, length, 1)
        
        # Transpose to match the expected input format of Conv1d
        # x = x.transpose(-1, -2).contiguous()  # Shape: (batch_size, 1, length)
        x = x.permute(0, 2, 1).contiguous()
        
        # Apply 1D convolution
        x = self.conv(x)  # Shape: (batch_size, 1, length)
        
        # Transpose back to the original format
        # x = x.transpose(-1, -2).contiguous()  # Shape: (batch_size, length, 1)
        x = x.permute(0, 2, 1).contiguous()
        
        # Squeeze and apply sigmoid
        x = torch.sigmoid(x)  # Shape: (batch_size, length, 1)

        # Element-wise multiplication with broadcasting
        return inputs * x  # Shape: (batch_size, length, channels) * (batch_size, length, 1)
    
class HeadDense(nn.Module):
    def __init__(self, input_dim, head_dim):
        super(HeadDense, self).__init__()
        self.dense = nn.Linear(input_dim, head_dim)
        self.activation = nn.SiLU()  # Equivalent to 'swish' in TensorFlow/Keras

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return x


class Conv1DBlockSqueezeformer(nn.Module):
    def __init__(self, channel_size, kernel_size, dilation_rate=1,
                 expand_ratio=4, activation='SiLU'):
        super(Conv1DBlockSqueezeformer, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.expand_ratio = expand_ratio
        self.activation = activation
        self.glu_layer = GLU()
        self.ffn = GLUMlp(self.channel_size * self.expand_ratio, self.channel_size)
        self.layer_norm_2 = nn.LayerNorm(channel_size, eps=1e-6)
        self.scale_bias_1 = ScaleBias(channel_size)
        self.scale_bias_2 = ScaleBias(channel_size)
        self.channel_size_dwconv = channel_size * expand_ratio // 2
        self.dwconv = nn.Conv1d(
            in_channels = self.channel_size_dwconv,
            out_channels = self.channel_size_dwconv,
            kernel_size = kernel_size,
            stride = 1,
            padding = 'same',
            dilation = dilation_rate,
            groups = self.channel_size_dwconv,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(self.channel_size_dwconv, momentum=0.95)
        self.conv_activation = nn.SiLU() if activation == 'SiLU' or activation == 'swish' else nn.ReLU()
        self.eca_layer = ECA()
        self.expand = nn.Linear(channel_size, channel_size * expand_ratio)
        self.project = nn.Linear(self.channel_size_dwconv, channel_size)

    def forward(self, x):
        skip = x
        x = self.expand(x)
        x = self.glu_layer(x)
        # x = self.dwconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dwconv(x.permute(0, 2, 1).contiguous())
        x = self.batch_norm(x).permute(0, 2, 1).contiguous()
        x = self.conv_activation(x)
        x = self.eca_layer(x)
        x = self.project(x)
        x = self.scale_bias_1(x)

        x = x + skip
        residual = x
        x = self.ffn(x)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x
    
# class Reshape1(nn.Module):
#     def __init__(self, col_len):
#         super(Reshape1, self).__init__()
#         self.col_len = col_len

#     def forward(self, x):
#         # First part of x
#         x_seq = x[:, :self.col_len]
#         x_seq = x_seq.view(-1, int(self.col_len / 60), 60).contiguous()
#         x_seq = x_seq.permute(0, 2, 1)
        
#         # Second part of x
#         x_seq_N = x[:, self.col_len:]
#         x_seq_N = x_seq_N.unsqueeze(1).repeat(1, 60, 1)
        
#         # Concatenate along the last dimension
#         x = torch.cat([x_seq, x_seq_N], dim=-1)
#         return x
    
# class Reshape2(nn.Module):
#     def __init__(self):
#         super(Reshape2, self).__init__()

#     def forward(self, x_pred, x_confidence):
#         x = x_pred
        
#         # Process x_pred
#         x_seq = x[:, :, :5]
#         x_seq = x_seq.permute(0, 2, 1).contiguous()
#         x_seq = x_seq.reshape(-1, 60 * 5)
#         x_seq_N = x[:, :, 5:]
#         x_seq_N = x_seq_N.mean(dim=1)
#         x1 = torch.cat([x_seq, x_seq_N], dim=-1)

#         x = x_confidence
        
#         # Process x_confidence
#         x_seq = x[:, :, :5]
#         x_seq = x_seq.permute(0, 2, 1).contiguous()
#         x_seq = x_seq.reshape(-1, 60 * 5)
#         x_seq_N = x[:, :, 5:]
#         x_seq_N = x_seq_N.mean(dim=1)
#         x2 = torch.cat([x_seq, x_seq_N], dim=-1)

#         x = torch.cat([x1, x2], dim=-1)
#         return x
#         # return x1