import numpy as np
import torch

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