import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from typing import List

"""
Contains the code for the resLSTM and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class PureResLSTMConfLossMetaData(modulus.ModelMetaData):
    name: str = "pure_resLSTM_conf_loss"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class PureResLSTMConfLoss(modulus.Module):
    def __init__(
            self,
            input_profile_num: int = 9, # number of input profile
            input_scalar_num: int = 17, # number of input scalars
            target_profile_num: int = 5, # number of target profile
            target_scalar_num: int = 8, # number of target scalars
            output_prune: bool = True, # whether or not we prune strato_lev_out levels
            strato_lev_out: int = 12, # number of levels to set to zero
            loc_embedding: bool = False, # whether or not to use location embedding
            embedding_type: str = "positional", # type of location embedding
            num_lstm: int = 10, # number of LSTM layers
            hidden_state: int = 256, # number of hidden units in LSTM
            ):

        super().__init__(meta=PureResLSTMConfLossMetaData())
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.loc_embedding = loc_embedding
        self.embedding_type = embedding_type
        self.num_lstm = num_lstm
        self.hidden_state = hidden_state
        self.vertical_level_num = 60
        self.inputs_dim = input_profile_num + input_scalar_num
        self.targets_dim = target_profile_num + target_scalar_num
        self.num_lstm = num_lstm

        residual_layers = nn.ModuleList()
        lstm_layers = nn.ModuleDict()
        for i in range(num_lstm):  
            lstm_key = f'lstm{i + 1}'
            lstm_layers[lstm_key] = nn.LSTM(input_size=self.inputs_dim if i == 0 else self.hidden_state*2,
                                            hidden_size=self.hidden_state, num_layers=1,
                                            bidirectional=True, batch_first=True)
            if i == 0:
                residual_layers.append(nn.Sequential(nn.LayerNorm([self.vertical_level_num, self.hidden_state*2]),nn.GELU()))
            else:
                residual_layers.append(nn.LayerNorm([self.vertical_level_num, self.hidden_state*2]),)

        self.res_act = nn.GELU()
        self.lstm_stack = lstm_layers
        self.residual_stack = residual_layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state*2, self.targets_dim*2),
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
                       
    def forward(self, x):
        profile_part = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1,self.input_profile_num,self.vertical_level_num).transpose(1,2)
        scalar_part = x[:,self.input_profile_num*self.vertical_level_num:].unsqueeze(1).repeat(1,self.vertical_level_num,1)
        inputs_seq = torch.cat([profile_part, scalar_part], dim = -1)
        outputs = inputs_seq  # b,60,self.inputs_dim
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

        outputs = self.fc(outputs)  # b,60,self.targets_dim

        outputs_pred = outputs[:,:,:self.targets_dim]
        outputs_conf = outputs[:,:,self.targets_dim:]

        profile_part_pred = outputs_pred[:,:,:self.target_profile_num]
        profile_part_pred = profile_part_pred.permute(0,2,1).reshape(-1,self.target_profile_num * self.vertical_level_num) # b,300
        scalar_part_pred = outputs_pred[:,:,self.target_profile_num:]
        scalar_part_pred = torch.mean(scalar_part_pred, dim=1) # b,8

        profile_part_conf = outputs_conf[:,:,:self.target_profile_num]
        profile_part_conf = profile_part_conf.permute(0,2,1).reshape(-1,self.target_profile_num * self.vertical_level_num)
        scalar_part_conf = outputs_conf[:,:,self.target_profile_num:]
        scalar_part_conf = torch.mean(scalar_part_conf, dim=1)

        y_pred = torch.concat([profile_part_pred, scalar_part_pred], dim=1)
        y_conf = torch.concat([profile_part_conf, scalar_part_conf], dim=1)

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