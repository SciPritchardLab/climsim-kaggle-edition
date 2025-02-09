import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
import nvtx
from layers import (
    FeatureScale,
    ResidualBlock,
)
from torch.nn.functional import silu
from typing import List

"""
Contains the code for the Pao model and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class pao_model_metadata(modulus.ModelMetaData):
    name: str = "pao_model"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class pao_model_nn(modulus.Module):
    def __init__(self,
                 input_series_num: int = 9, # number of input series
                 input_scalar_num: int = 17, # number of input scalar
                 target_series_num: int = 5, # number of target series
                 target_scalar_num: int = 8, # number of target scalar
                 hidden_series_num: int = 160, # number of hidden units in MLP for series
                 hidden_scalar_num: int = 160, # number of hidden units in MLP for scalar
                 output_prune: bool = True, # whether or not we prune strato_lev_out levels
                 strato_lev_out: int = 12, # number of levels to set to zero
                ):
        super().__init__(meta = pao_model_metadata())
        self.input_series_num = input_series_num
        self.input_scalar_num = input_scalar_num
        self.target_series_num = target_series_num
        self.target_scalar_num = target_scalar_num
        self.hidden_series_num = hidden_series_num
        self.hidden_scalar_num = hidden_scalar_num
        self.num_hidden_total = self.hidden_series_num + self.hidden_scalar_num
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        # 60 series 1d cnn
        self.feature_scale = nn.ModuleList([
            FeatureScale(60) for _ in range(self.input_series_num)
        ])
        self.positional_encoding = nn.Embedding(60, self.hidden_series_num)
        self.input_linear = nn.Linear(self.input_series_num, self.hidden_series_num)  # current, diff
        self.other_feats_mlp = nn.Sequential(
            nn.Linear(self.input_scalar_num, self.hidden_scalar_num),
            nn.BatchNorm1d(self.hidden_scalar_num),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.hidden_scalar_num, self.hidden_scalar_num),
            nn.BatchNorm1d(self.hidden_scalar_num),
            # nn.ReLU(),
            nn.GELU()
        )
        self.other_feats_proj = nn.ModuleList([nn.Linear(self.hidden_scalar_num, self.hidden_scalar_num) for _ in range(60)])
        # layer norm
        self.layer_norm_in = self.num_hidden_total
        self.series_layer_norm = nn.LayerNorm(self.layer_norm_in)
        # cnn
        self.cnn1 = nn.Sequential(
            ResidualBlock(self.num_hidden_total, self.num_hidden_total, 5, 1, 2),
            ResidualBlock(self.num_hidden_total, self.num_hidden_total, 5, 1, 2),
            ResidualBlock(self.num_hidden_total, self.num_hidden_total, 5, 1, 2),
            ResidualBlock(self.num_hidden_total, self.num_hidden_total, 5, 1, 2)
        )
        # lstm
        self.lstm = nn.Sequential(
            nn.LSTM(self.num_hidden_total, self.num_hidden_total, 2, batch_first=True, bidirectional=True, dropout=0.0),
        )
        # output layer
        self.output_series_mlp_input_dim = self.num_hidden_total * 2
        self.series_output_mlp = nn.Sequential(
            nn.Linear(self.output_series_mlp_input_dim, self.num_hidden_total),
            nn.GELU(),
            nn.Linear(self.num_hidden_total, self.num_hidden_total),
            nn.GELU(),
            nn.Linear(self.num_hidden_total, self.hidden_series_num),
            nn.GELU(),
            nn.Linear(self.hidden_series_num, self.target_series_num)
        )
        self.output_scalar_mlp_input_dim = self.num_hidden_total * 60 * 2
        self.scalar_layer_norm = nn.LayerNorm(self.output_scalar_mlp_input_dim)
        self.scalar_output_mlp = nn.Sequential(
            nn.Linear(self.output_scalar_mlp_input_dim, self.hidden_scalar_num),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.hidden_scalar_num, self.hidden_scalar_num),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.hidden_scalar_num, self.target_scalar_num)
        )

    def reshape_input(self, x):
        batch_size = x.size(0)
        series_part = x[:, :self.input_series_num*60]
        series_inputs = series_part.reshape(batch_size, self.input_series_num, 60)
        scalar_inputs = x[:, self.input_series_num*60:]
        return series_inputs, scalar_inputs

    def reshape_output(self, series_output, scalar_output):
        batch_size = series_output.size(0)
        series_part = series_output.reshape(batch_size, self.target_series_num * 60)
        return torch.cat([series_part, scalar_output], dim = 1)
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        series_inputs, scalar_inputs = self.reshape_input(x)
        # create series_features
        # create scalar_inputs
        # concat dim60 cols
        dim60_x = []
        scale_ix = 0
        for group_ix in range((self.input_series_num)):
            origin_x = series_inputs[:, group_ix, :] # (batch, 60)
            x = self.feature_scale[scale_ix](origin_x)  # (batch, 60)
            scale_ix += 1
            x = x.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x)
            # # diff feature
            # x_diff = origin_x[:, 1:] - origin_x[:, :-1]  # (batch, 59)
            # x_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 1), x_diff], dim=1)  # (batch, 60)
            # x_diff = self.feature_scale[scale_ix](x_diff)  # (batch, 60)
            # scale_ix += 1
            # x_diff = x_diff.unsqueeze(-1)  # (batch, 60, 1)
            # dim60_x.append(x_diff)
            # # diff diff feature
            # x_diff = origin_x[:, 1:] - origin_x[:, :-1]
            # x_diff_diff = x_diff[:, 1:] - x_diff[:, :-1]  # (batch, 58)
            # x_diff_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 2), x_diff_diff], dim=1)  # (batch, 60)
            # x_diff_diff = self.feature_scale[scale_ix](x_diff_diff)  # (batch, 60)
            # scale_ix += 1
            # x_diff_diff = x_diff_diff.unsqueeze(-1)  # (batch, 60, 1)
            # dim60_x.append(x_diff_diff)

        x = torch.cat(dim60_x, dim=2)  # (batch, 60, self.input_series_num)
        position = torch.arange(0, 60, device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (x.size(0)->batch, 60)
        position = self.positional_encoding(position)  # (batch, 60, 128)
        x = self.input_linear(x)  # (batch, series_len, 128)
        x = x + position
        # other cols
        scalar_x = scalar_inputs  # (batch, 19)
        scalar_x = self.other_feats_mlp(scalar_x)  # (batch, hidden)
        scalar_x_list = []
        for i in range(60):
            scalar_x_list.append(self.other_feats_proj[i](scalar_x))
        scalar_x = torch.stack(scalar_x_list, dim=1)  # (batch, 60, hidden)
        # repeat to match series_len
        # scaler_x = scaler_x.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, series_len, hidden)
        # concat
        x = torch.cat([x, scalar_x], dim=2)  # (batch, 60, hidden*2)
        x = self.series_layer_norm(x)

        x = x.transpose(1, 2)  # (batch, hidden, series_len)
        x = self.cnn1(x)  # (batch, hidden, series_len)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)  # (batch, series_len, hidden)

        series_output = self.series_output_mlp(x)  # (batch, series_len, n_targets)
        # series_diff_output = series_output[:, :, self.target_series_num:]
        # series_output = series_output[:, :, :self.target_series_num]
        x = x.reshape(x.size(0), -1)
        x = self.scalar_layer_norm(x)
        scalar_output = self.scalar_output_mlp(x)
        y = self.reshape_output(series_output, scalar_output)
        if self.output_prune:
            # Zeyuan says that the .clone() and .clone().zero_() helped bypass a torchscript issue. Reason unclear.
            y = y.clone()
            y[:, 60:60+self.strato_lev_out] = y[:, 60:60+self.strato_lev_out].clone().zero_()
            y[:, 120:120+self.strato_lev_out] = y[:, 120:120+self.strato_lev_out].clone().zero_()
            y[:, 180:180+self.strato_lev_out] = y[:, 180:180+self.strato_lev_out].clone().zero_()
            y[:, 240:240+self.strato_lev_out] = y[:, 240:240+self.strato_lev_out].clone().zero_()
        return y

if __name__ == '__main__':
    model = pao_model()
    print(model.count_trainable_parameters())