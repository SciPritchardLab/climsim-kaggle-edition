import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
import nvtx
from torch.nn.functional import silu
from typing import List

"""
Contains the code for the resLSTM and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class resLSTM_metadata(modulus.ModelMetaData):
    name: str = "resLSTM"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class FeatureScale(nn.Module):
    def __init__(self, input_dim):
        super(FeatureScale, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.biases = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        # 各次元ごとに対応するパラメータを掛ける
        return x * self.weights + self.biases


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_features),
            # nn.ReLU(),
            nn.GELU(),
            nn.Conv1d(out_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_features),
            # nn.ReLU()
            nn.GELU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(out_features, nhead=8, dim_feedforward=out_features, dropout=0.0, batch_first=True),
            num_layers=1,
        )


    def forward(self, x):
        out = self.conv(x)
        out = out + x
        out = out.transpose(1, 2)
        out = self.transformer(out)
        out = out.transpose(1, 2)
        return out


class LeapModel(nn.Module):
    def __init__(self):
        super(LeapModel, self).__init__()
        # 60 sequences 1d cnn
        self.feature_scale = nn.ModuleList([
            FeatureScale(60) for _ in range((len(FEATURE_SEQ_GROUPS)+3) * 3 * 1)
        ])
        self.positional_encoding = nn.Embedding(60, 128)
        self.input_linear = nn.Linear((len(FEATURE_SEQ_GROUPS)+3) * 3 * 1, 128)  # current, diff
        n_hidden = 128
        self.other_feats_mlp = nn.Sequential(
            nn.Linear(len(FEATURE_SCALER_COLS), n_hidden),
            nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            nn.GELU()
        )
        self.other_feats_proj = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(60)])
        # layer norm
        layer_norm_in = n_hidden + 128
        self.seq_layer_norm = nn.LayerNorm(layer_norm_in)
        # cnn
        n_cnn_hidden = n_hidden + 128
        self.cnn1 = nn.Sequential(
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2)
        )
        # lstm
        self.lstm = nn.Sequential(
            nn.LSTM(n_cnn_hidden, n_cnn_hidden, 2, batch_first=True, bidirectional=True, dropout=0.0),
        )
        # output layer
        output_seq_mlp_input_dim = n_cnn_hidden*2
        self.seq_output_mlp = nn.Sequential(
            nn.Linear(output_seq_mlp_input_dim, n_hidden*2),
            nn.GELU(),
            nn.Linear(n_hidden*2, n_hidden*2),
            nn.GELU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, len(TARGET_SEQ_GROUPS) * 2)
        )
        output_scaler_mlp_input_dim = n_cnn_hidden*2*60
        self.scaler_layer_norm = nn.LayerNorm(output_scaler_mlp_input_dim)
        self.scaler_output_mlp = nn.Sequential(
            nn.Linear(output_scaler_mlp_input_dim, n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, len(TARGET_SCALER_COLS))
        )

    def forward(self, scaler_features, seq_features):
        # concat dim60 cols
        dim60_x = []
        scale_ix = 0
        # for group_ix in range(len(FEATURE_SEQ_GROUPS) * 2):
        for group_ix in range((len(FEATURE_SEQ_GROUPS)+3) * 1):
            origin_x = seq_features[:, group_ix, :]
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
        position = torch.arange(0, 60, device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (batch, 60)
        position = self.positional_encoding(position)  # (batch, 60, 16)
        x = self.input_linear(x)  # (batch, seq_len, 128)
        x = x + position
        # other cols
        scaler_x = scaler_features  # (batch, n_feats)
        scaler_x = self.other_feats_mlp(scaler_x)  # (batch, hidden)
        scaler_x_list = []
        for i in range(60):
            scaler_x_list.append(self.other_feats_proj[i](scaler_x))
        scaler_x = torch.stack(scaler_x_list, dim=1)  # (batch, 60, hidden)
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
        seq_diff_output = seq_output[:, :, len(TARGET_SEQ_GROUPS):]
        seq_output = seq_output[:, :, :len(TARGET_SEQ_GROUPS)]
        # scaler_head
        scaler_x = x.reshape(x.size(0), -1)
        scaler_x = self.scaler_layer_norm(scaler_x)
        scaler_output = self.scaler_output_mlp(scaler_x)
        return seq_output, scaler_output, seq_diff_output


def train_one_epoch(model, loss_fn, data_loader, optimizer,
                    device, scheduler, epoch, scaler=None, awp=None, ema=None):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)

    model.train()

    bar = tqdm(enumerate(data_loader), total=len(data_loader))

    scaler_weight_arr = np.asarray([new_weight[c] for c in TARGET_SCALER_COLS])
    seq_weight_arr = np.asarray([[new_weight[f"{c}_{i}"] for c in TARGET_SEQ_GROUPS] for i in range(60)])
    scaler_weight_mask = np.where(scaler_weight_arr == 0, 0, 1)
    seq_weight_mask = np.where(seq_weight_arr == 0, 0, 1)
    seq_weight_mask[12:27, TARGET_SEQ_GROUPS.index("ptend_q0002")] = 0.0
    scaler_weight_mask = torch.tensor(scaler_weight_mask, dtype=torch.float).to(device)
    seq_weight_mask = torch.tensor(seq_weight_mask, dtype=torch.float).to(device)

    for iter_i, batch in bar:
        # input
        seq_features = batch["seq_features"].to(device)
        scaler_features = batch["scaler_features"].to(device)
        seq_targets = batch["seq_targets"].to(device)
        scaler_targets = batch["scaler_targets"].to(device)
        seq_targets_diff = batch["seq_targets_diff"].to(device)
        batch_size = len(scaler_targets)

        # zero grad
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            with amp.autocast(enabled=CFG.use_fp16):
                seq_preds, scaler_preds, seq_diff_preds = model(scaler_features, seq_features)
                # mask
                seq_preds = seq_preds * seq_weight_mask
                scaler_preds = scaler_preds * scaler_weight_mask
                seq_diff_preds = seq_diff_preds * seq_weight_mask
                scaler_targets = scaler_targets * scaler_weight_mask
                seq_targets = seq_targets * seq_weight_mask
                seq_targets_diff = seq_targets_diff * seq_weight_mask
                # loss function
                seq_loss = loss_fn(seq_preds.reshape(-1), seq_targets.reshape(-1))
                scaler_loss = loss_fn(scaler_preds.reshape(-1), scaler_targets.reshape(-1))
                seq_diff_loss = loss_fn(seq_diff_preds.reshape(-1), seq_targets_diff.reshape(-1))
                loss = seq_loss + scaler_loss + seq_diff_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            epoch_loss += loss.item()
        
        bar.set_postfix(OrderedDict(loss=loss.item(), lr=optimizer.param_groups[0]['lr']))
    
    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data