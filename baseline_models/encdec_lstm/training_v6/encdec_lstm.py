import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    MLP,
)
from typing import List
"""
Contains the code for the ConvNeXt model
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class EncDecLSTMMetaData(modulus.ModelMetaData):
    name: str = "encdec_lstm"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False

class EncDecLSTM(modulus.Module):
    def __init__(self,
                 input_profile_num: int = 9, # number of input profile variables
                 input_scalar_num: int = 17, # number of input scalar variables
                 target_profile_num: int = 5, # number of target profile variables
                 target_scalar_num: int = 8, # number of target scalar variables
                 output_prune: bool = True, # whether or not we prune strato_lev_out levels
                 strato_lev_out: int = 12, # number of levels to set to zero
                 loc_embedding: bool = False, # whether or not to use location embedding
                 embedding_type: str = "positional", # type of location embedding
                 hidden_dim: int = 512, # hidden dimension of LSTM
                 num_layers: int = 3, # number of LSTM layers
                ):
        super().__init__(meta = EncDecLSTMMetaData())
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.vertical_level_num = 60
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.loc_embedding = loc_embedding
        self.embedding_type = embedding_type
        self.num_input_vars = self.input_profile_num + self.input_scalar_num
        self.output_dim = self.target_profile_num * self.vertical_level_num + self.target_scalar_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = MLP([self.num_input_vars * self.vertical_level_num, self.num_input_vars * 15, self.num_input_vars * 8, self.num_input_vars * 4])
        self.decoder = MLP([self.num_input_vars * 4, self.num_input_vars * 8, self.num_input_vars * 15, self.num_input_vars * self.vertical_level_num])

        self.lstm = nn.LSTM(input_size = self.num_input_vars * 2, \
                            hidden_size = self.hidden_dim, \
                            num_layers = self.num_layers,
                            batch_first=True, dropout=0.09, bidirectional=True)
        
        self.lstm2 = nn.GRU(input_size = self.hidden_dim * 2, \
                            hidden_size = 16, batch_first=True, dropout=0.00, bidirectional=True)

        self.fc_lstm = MLP([32 * 60, 16 * 60, self.output_dim])


    def forward(self, x):
        # reshape input
        profile_part = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1,self.input_profile_num,self.vertical_level_num).transpose(1,2)
        scalar_part = x[:,self.input_profile_num*self.vertical_level_num:].unsqueeze(1).repeat(1,self.vertical_level_num,1)
        x = torch.cat([profile_part, scalar_part], dim = -1) # b, 60, 26
        # flatten to b, 60 * 26
        x_enc = self.encoder(x.reshape(-1, self.num_input_vars * self.vertical_level_num))
        x_dec = self.decoder(x_enc).reshape(-1, self.vertical_level_num, self.num_input_vars)

        lstm_out, _ = self.lstm(torch.cat([x, x_dec], dim = -1))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.reshape(-1, 32 * 60)
        y = self.fc_lstm(lstm_out) # b, 308

        if self.output_prune:
            # Zeyuan says that the .clone() and .clone().zero_() helped bypass a torchscript issue. Reason unclear.
            y = y.clone()
            y[:, 60:60+self.strato_lev_out] = y[:, 60:60+self.strato_lev_out].clone().zero_()
            y[:, 120:120+self.strato_lev_out] = y[:, 120:120+self.strato_lev_out].clone().zero_()
            y[:, 180:180+self.strato_lev_out] = y[:, 180:180+self.strato_lev_out].clone().zero_()
            y[:, 240:240+self.strato_lev_out] = y[:, 240:240+self.strato_lev_out].clone().zero_()
        return y