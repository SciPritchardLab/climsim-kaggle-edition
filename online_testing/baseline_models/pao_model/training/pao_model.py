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

class pao_model(modulus.Module):
    def __init__(self,
                 num_seq_inputs: int = 23, # number of input sequences
                 num_scalar_inputs: int = 19, # number of input scalars
                 num_seq_targets: int = 5, # number of target sequences
                 num_scalar_targets: int = 8, # number of target scalars
                 num_hidden: int = 128, # number of hidden units in MLP
                ):
        super().__init__(meta = pao_model_metadata())
        self.num_seq_inputs = num_seq_inputs
        self.num_scalar_inputs = num_scalar_inputs
        self.num_seq_targets = num_seq_targets
        self.num_scalar_targets = num_scalar_targets
        self.num_hidden = num_hidden
        # 60 sequences 1d cnn
        self.feature_scale = nn.ModuleList([
            FeatureScale(60) for _ in range((self.num_seq_inputs+3) * 3 * 1)
        ])
        self.positional_encoding = nn.Embedding(60, 128)
        self.input_linear = nn.Linear((self.num_seq_inputs+3) * 3 * 1, 128)  # current, diff
        self.other_feats_mlp = nn.Sequential(
            nn.Linear(self.num_seq_groups, self.num_hidden),
            nn.BatchNorm1d(self.num_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.BatchNorm1d(self.num_hidden),
            # nn.ReLU(),
            nn.GELU()
        )
        self.other_feats_proj = nn.ModuleList([nn.Linear(self.num_hidden, self.num_hidden) for _ in range(60)])
        # layer norm
        self.layer_norm_in = self.num_hidden * 2
        self.seq_layer_norm = nn.LayerNorm(self.layer_norm_in)
        # cnn
        self.num_cnn_hidden = self.num_hidden * 2
        self.cnn1 = nn.Sequential(
            ResidualBlock(self.num_cnn_hidden, self.num_cnn_hidden, 5, 1, 2),
            ResidualBlock(self.num_cnn_hidden, self.num_cnn_hidden, 5, 1, 2),
            ResidualBlock(self.num_cnn_hidden, self.num_cnn_hidden, 5, 1, 2),
            ResidualBlock(self.num_cnn_hidden, self.num_cnn_hidden, 5, 1, 2)
        )
        # lstm
        self.lstm = nn.Sequential(
            nn.LSTM(self.num_cnn_hidden, self.num_cnn_hidden, 2, batch_first=True, bidirectional=True, dropout=0.0),
        )
        # output layer
        self.output_seq_mlp_input_dim = self.num_cnn_hidden*2
        self.seq_output_mlp = nn.Sequential(
            nn.Linear(self.output_seq_mlp_input_dim, self.num_hidden*2),
            nn.GELU(),
            nn.Linear(self.num_hidden*2, self.num_hidden*2),
            nn.GELU(),
            nn.Linear(self.num_hidden*2, self.num_hidden),
            nn.GELU(),
            nn.Linear(self.num_hidden, self.num_seq_targets * 2)
        )
        self.output_scalar_mlp_input_dim = self.num_cnn_hidden*2*60
        self.scalar_layer_norm = nn.LayerNorm(self.output_scalar_mlp_input_dim)
        self.scalar_output_mlp = nn.Sequential(
            nn.Linear(self.output_scalar_mlp_input_dim, self.num_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self.num_hidden, self.num_scalar_targets)
        )

    def forward(self, seq_features, scalar_features):
        # concat dim60 cols
        dim60_x = []
        scale_ix = 0
        # for group_ix in range(len(FEATURE_SEQ_GROUPS) * 2):
        for group_ix in range((self.num_seq_inputs+3) * 1):
            origin_x = seq_features[:, group_ix, :] # (batch, 60)
            x = self.feature_scale[scale_ix](origin_x)  # (batch, 60)
            scale_ix += 1
            x = x.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x)
            # diff feature
            x_diff = origin_x[:, 1:] - origin_x[:, :-1]  # (batch, 59)
            x_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 1), x_diff], dim=1)  # (batch, 60)
            x_diff = self.feature_scale[scale_ix](x_diff)  # (batch, 60)
            scale_ix += 1
            x_diff = x_diff.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x_diff)
            # diff diff feature
            x_diff = origin_x[:, 1:] - origin_x[:, :-1]
            x_diff_diff = x_diff[:, 1:] - x_diff[:, :-1]  # (batch, 58)
            x_diff_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 2), x_diff_diff], dim=1)  # (batch, 60)
            x_diff_diff = self.feature_scale[scale_ix](x_diff_diff)  # (batch, 60)
            scale_ix += 1
            x_diff_diff = x_diff_diff.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x_diff_diff)

        x = torch.cat(dim60_x, dim=2)  # (batch, 60, M)
        position = torch.arange(0, 60, device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (x.size(0)->batch, 60)
        position = self.positional_encoding(position)  # (batch, 60, 128)
        x = self.input_linear(x)  # (batch, seq_len, 128)
        x = x + position
        # other cols
        scalar_x = scalar_features  # (batch, n_feats)
        scalar_x = self.other_feats_mlp(scalar_x)  # (batch, hidden)
        scalar_x_list = []
        for i in range(60):
            scalar_x_list.append(self.other_feats_proj[i](scalar_x))
        scalar_x = torch.stack(scalar_x_list, dim=1)  # (batch, 60, hidden)
        # repeat to match seq_len
        # scaler_x = scaler_x.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, hidden)
        # concat
        x = torch.cat([x, scaler_x], dim=2)  # (batch, seq_len, hidden*2)
        x = self.seq_layer_norm(x)

        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.cnn1(x)  # (batch, hidden, seq_len)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # seq_head
        seq_output = self.seq_output_mlp(x)  # (batch, seq_len * 2, n_targets)
        seq_diff_output = seq_output[:, :, self.num_seq_targets:]
        seq_output = seq_output[:, :, :self.num_seq_targets]
        # scalar_head
        scalar_x = x.reshape(x.size(0), -1)
        scalar_x = self.scalar_layer_norm(scaler_x)
        scalar_output = self.scalar_output_mlp(scaler_x)
        return seq_output, scalar_output, seq_diff_output