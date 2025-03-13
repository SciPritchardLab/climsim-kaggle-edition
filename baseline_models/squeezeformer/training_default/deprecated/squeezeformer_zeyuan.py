import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import Reshape1, Conv1DBlockSqueezeformer, TransformerEncoder, HeadDense, GLUMlp, Reshape2
"""
Contains the code for the Sqeezeformer and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SqueezeformerMetaData(modulus.ModelMetaData):
    name: str = "Squeezeformer"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = False
    
    
class Squeezeformer(modulus.Module):
    """
    Sqeezeformer Estimator
    """
    def __init__(self, col_len, col_not_len, dim=256, head_dim=1024):
        super().__init__(meta=SqueezeformerMetaData())
        self.reshape1 = Reshape1(col_len)
        self.dense = nn.Linear(int(col_len/60) + col_not_len, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.conv_blocks = nn.ModuleList([Conv1DBlockSqueezeformer(dim, 15) for _ in range(12)])
        self.transformer_encoders = nn.ModuleList([TransformerEncoder(dim, 4, dim * 4) for _ in range(12)])
        self.head_dense = HeadDense(dim, head_dim)
        self.ffn = GLUMlp(head_dim * 2, head_dim)
        self.final_dense_pred = nn.Linear(head_dim, 13)
        self.final_dense_confidence = nn.Linear(head_dim, 13) # 20 was for auxilary loss
        self.reshape2 = Reshape2()

    def forward(self, x):
        x = self.reshape1(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        for conv_block, transformer_encoder in zip(self.conv_blocks, self.transformer_encoders):
            x = conv_block(x)
            x = transformer_encoder(x)
        # for conv_block in self.conv_blocks:
        #     x = conv_block(x)

        x = self.head_dense(x)
        x = self.ffn(x)

        x_pred = self.final_dense_pred(x)
        x_confidence = self.final_dense_confidence(x)

        x = self.reshape2(x_pred, x_confidence)
        return x