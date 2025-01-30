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

class resLSTM(modulus.Module):
    def __init__(
            self,
            inputs_dim: int = 42, # number of sequences
            num_lstm: int = 10,
            hidden_state: int = 512,
            ):
        
        super().__init__(meta=resLSTM_metadata())
        self.inputs_dim = inputs_dim
        self.num_lstm = num_lstm
        self.hidden_state = hidden_state
        self.output_single_num = 8

        residual_layers = nn.ModuleList()
        lstm_layers = nn.ModuleDict()
        for i in range(num_lstm):  
            lstm_key = f'lstm{i + 1}'
            lstm_layers[lstm_key] = nn.LSTM(input_size=self.inputs_dim if i == 0 else self.hidden_state*2,
                                            hidden_size=self.hidden_state, num_layers=1,
                                            bidirectional=True, batch_first=True)
            if i == 0:
                residual_layers.append(nn.Sequential(nn.LayerNorm([60, self.hidden_state*2]),nn.GELU()))
            else:
                residual_layers.append(nn.LayerNorm([60, self.hidden_state*2]),)

        self.res_act = nn.GELU()
        self.lstm_stack = lstm_layers
        self.residual_stack = residual_layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state*2, 13),
        )
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
                       
    def forward(self, inputs):
        outputs = inputs
        last_outputs = outputs
        # pass through LSTM layer by layer and apply residual connection
        for i, (lstm, residual) in enumerate(zip(self.lstm_stack.values(), self.residual_stack)):
            outputs, _ = lstm(outputs)
            outputs = residual(outputs)
            if i > 0:
                outputs = self.res_act(0.7*outputs+0.3*last_outputs)  # residual connection
                last_outputs = outputs
            else:  # i % 2 ==0
                last_outputs = outputs  # save predictions of last two layers

        outputs = self.fc(outputs)  # b,60,13

        series_part = outputs[:,:,self.output_single_num:]
        series_part = series_part.permute(0,2,1).reshape(-1,300) # b,300
        single_part = outputs[:,:,:self.output_single_num]
        single_part = torch.mean(single_part, dim=1) # b,8

        outputs = torch.concat([series_part, single_part], dim=1)
        return outputs