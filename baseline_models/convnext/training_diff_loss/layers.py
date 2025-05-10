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

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(DownsampleLayer, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride = 1, padding = 'same')

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

class PointWiseLinear(nn.Module):
    def __init__(self, dim):
        super(PointWiseLinear, self).__init__()
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pwconv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act(x)
        x = self.pwconv2(x.permute(0, 2, 1))
        return x

# https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py
class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        pointwise=PointWiseLinear,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding="same", stride=1, groups=dim
        )  # depthwise conv
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.pwconv = pointwise(dim=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = input + self.drop_path(x)
        return x

class DoubleBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        pointwise=PointWiseLinear,
        kernel_size_60x60=1,
    ):
        super().__init__()
        self.convkx1 = nn.Conv1d(
            dim,
            dim // 2,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            groups=dim // 2,
        )  # depthwise conv
        self.act = nn.GELU()

        self.conv1xk = nn.Conv1d(60, 60, kernel_size=4, padding=1, stride=2, groups=60)
        self.normkx1 = nn.BatchNorm1d(dim // 2)
        self.norm1xk = nn.BatchNorm1d(60)
        self.pwconv = pointwise(dim=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x1 = self.convkx1(x)  # shape = (bs, dim, 60) -> (bs, dim//2, 60)
        x1 = self.normkx1(x1)
        x2 = x.permute(0, 2, 1)
        x2 = self.conv1xk(x2)  # shape = (bs, 60, dim) -> (bs, 60, dim//2)
        x2 = self.norm1xk(x2)  # shape = (bs, 60, dim) -> (bs, 60, dim//2)
        x2 = x2.permute(0, 2, 1)

        x = torch.cat([x1, x2], dim=1)  # shape = (bs, dim, 60)

        x = self.pwconv(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # shape = (bs, dim, 60)
        x = input + self.drop_path(x)
        return x