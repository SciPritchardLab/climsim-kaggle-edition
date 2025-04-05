import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    Conv1d,
    GroupNorm,
    Linear,
    UNetBlock,
    UNetBlock_noatten,
    UNetBlock_atten,
    ScriptableAttentionOp,
)
from torch.nn.functional import silu
from typing import List

"""
Contains the code for the Unet and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class UnetMetaData(modulus.ModelMetaData):
    name: str = "unet"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False

class Unet(modulus.Module):
    def __init__(
            self, 
            input_profile_num: int = 9, # number of input profile variables
            input_scalar_num: int = 17, # number of input scalar variables
            target_profile_num: int = 5, # number of target profile variables
            target_scalar_num: int = 8, # number of target scalar variables
            output_prune: bool = True,
            strato_lev_out: int = 12,
            dropout: float = 0.0,
            loc_embedding: bool = False,
            embedding_type: str = "positional",
            num_blocks: int = 2,
            attn_resolutions: List[int] = [0],
            model_channels: int = 128,
            skip_conv: bool = False,
            prev_2d: bool = False,
            seq_resolution: int = 64,
            label_dim: int = 0,
            augment_dim: int = 0,
            channel_mult: List[int] = [1, 2, 2, 2],
            channel_mult_emb: int = 4,
            label_dropout: float = 0.0,
            channel_mult_noise: int = 1,
            encoder_type: str = "standard",
            decoder_type: str = "standard",
            resample_filter: List[int] = [1, 1],
            ):
        
        super().__init__(meta=UnetMetaData())
        # check if hidden_dims is a list of hidden_dims
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.model_channels = model_channels

        self.in_channels = input_profile_num + input_scalar_num
        self.out_channels = target_profile_num + target_scalar_num
        # print('1: out_channels', self.out_channels)

        # valid_encoder_types = ["standard", "skip", "residual"]
        valid_encoder_types = ["standard"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        # valid_decoder_types = ["standard", "skip"]
        valid_decoder_types = ["standard"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        self.label_dropout = label_dropout
        self.embedding_type = embedding_type

        self.seq_resolution = seq_resolution
        self.label_dim = label_dim
        self.augment_dim = augment_dim
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.channel_mult_emb = channel_mult_emb
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.channel_mult_noise = channel_mult_noise
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.resample_filter = resample_filter
        self.vertical_level_num = 60
        self.input_padding = (seq_resolution-self.vertical_level_num,0)
        self.output_prune=output_prune
        self.strato_lev_out=strato_lev_out
        self.loc_embedding = loc_embedding
        self.skip_conv = skip_conv
        self.prev_2d = prev_2d

        # emb_channels = model_channels * channel_mult_emb
        # self.emb_channels = emb_channels
        # noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=0.2**0.5)
        block_kwargs = dict(
            # emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=0.5**0.5,
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = self.in_channels # number of variables
        caux = self.in_channels # number of variables

        #channel_mult = [1, 2, 2, 2]
        for level, mult in enumerate(channel_mult):
            #decreases the resolution by a bit each level
            res = seq_resolution >> level

            if level == 0:#if its the first level
                cin = cout
                cout = model_channels # model_channels = 128
                # comment out the first conv layer that supposed to be the input embedding
                # because we will have the input embedding manusally for profile vars and scalar vars
                self.enc[f"{res}_conv"] = Conv1d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                #every multilayer section ends with a downsample block
                self.enc[f"{res}_down"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )

                #only current possible encoder type is standard
                if encoder_type == "skip":
                    self.enc[f"{res}_aux_down"] = Conv1d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}_aux_skip"] = Conv1d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}_aux_residual"] = Conv1d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout

            #for all 4 layers for this block
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult

                #if your resolution is the resolution value we set (8), then make an attention block
                attn = res in attn_resolutions

                if attn:
                    self.enc[f"{res}_block{idx}"] = UNetBlock_atten(
                        in_channels=cin, 
                        out_channels=cout, 
                        emb_channels=0,
                        up=False,
                        down=False,
                        channels_per_head=64,
                        **block_kwargs
                    )
                else:
                    # make a normal res block
                    self.enc[f"{res}_block{idx}"] = UNetBlock_noatten(
                        in_channels=cin, 
                        out_channels=cout, 
                        attention=attn,
                        emb_channels=0,
                        up=False,
                        down=False,
                        channels_per_head=64,
                        **block_kwargs
                    )
        
        # since we are using a standard encoder, there are no aux blocks
        # each res block has a skip connection and skips stores the out_channel for each of these
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        #creates skip layers
        self.skip_conv_layer = [] #torch.nn.ModuleList()
        # for each skip connection, add a 1x1 conv layer initialized as identity connection, with an option to train the weight
        for idx, skip in enumerate(skips):
            conv = Conv1d(in_channels=skip, out_channels=skip, kernel=1)
            torch.nn.init.dirac_(conv.weight)
            torch.nn.init.zeros_(conv.bias)
            if not self.skip_conv:
                conv.weight.requires_grad = False
                conv.bias.requires_grad = False
            self.skip_conv_layer.append(conv)
        
        self.skip_conv_layer = torch.nn.ModuleList(self.skip_conv_layer)
            # XX doulbe check if the above is correct

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        self.dec_aux_norm = torch.nn.ModuleDict()
        self.dec_aux_conv = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            #decreases the resolution for each level (starts high since order of layers is inverted, we start w outer layer)
            res = seq_resolution >> level

            #if this is the middle of the architecture, first part of decoder
            if level == len(channel_mult) - 1:
                self.dec[f"{res}_in0"] = UNetBlock_atten(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}_in1"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                #upsample block for each multilayer
                self.dec[f"{res}_up"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )

            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                if attn:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_atten(
                        in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                    )
                else:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_noatten(
                        in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                    )
            if decoder_type == "skip" or level == 0:
                # if decoder_type == "skip" and level < len(channel_mult) - 1:
                #     self.dec[f"{res}_aux_up"] = Conv1d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel=0,
                #         up=True,
                #         resample_filter=resample_filter,
                #     )
                self.dec_aux_norm[f"{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                ## comment out the last conv layer that supposed to recover the output channels
                ## we will manually recover the output channels
                self.dec_aux_conv[f"{res}_aux_conv"] = Conv1d(
                    in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero
                )

        # create a 385x8 trainable weight embedding for the input
                       
    def forward(self, x):
        '''
        x: (batch, input_profile_num*levels+input_scalar_num)
        # x_profile: (batch, input_profile_num, levels)
        # x_scalar: (batch, input_scalar_num)
        '''

        # if self.qinput_prune:
        #     x = x.clone()  # Clone the tensor to ensure you're not modifying the original tensor in-place
        #     x[:, 60:60+self.strato_lev] = x[:, 60:60+self.strato_lev].clone().zero_()  # Set stratosphere q1 to 0
        #     x[:, 120:120+self.strato_lev] = x[:, 120:120+self.strato_lev].clone().zero_()  # Set stratosphere q2 to 0
        #     x[:, 180:180+self.strato_lev] = x[:, 180:180+self.strato_lev].clone().zero_()  # Set stratosphere q3 to 0

        # if not self.prev_2d:
        #     x = x.clone()
        #     x[:,-8:-3] = x[:,-8:-3].clone().zero_()

        # split x into x_profile and x_scalar
        x_profile = x[:,:self.input_profile_num*self.vertical_level_num]
        x_scalar = x[:,self.input_profile_num*self.vertical_level_num:]


        # print(x_profile.shape, x_scalar.shape, x_loc.shape)

        # reshape x_profile to (batch, input_profile_num, levels)
        x_profile = x_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
        # broadcast x_scalar to (batch, input_scalar_num, levels)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)

        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
        x = torch.cat((x_profile, x_scalar), dim=1)
        # print('2:', x.shape)
        # x = torch.cat((x_profile, x_scalar), dim=1)
        
        # pads the beginning of levels so that levels = seq_resolution (which by default is 64)
        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        # print('3:', x.shape)
        # pass the concatenated tensor through the Unet

        # Encoder.
        skips = []
        aux = x
        #loops through all the values in the encoder dictionary
        for name, block in self.enc.items():
            #runs the data through the proper block
            # since we are using a standard encoder, there are no aux blocks in the encoder
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                #replaces the last skip block
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                #replaces the last skip block and updates the current block
                x = skips[-1] = aux = (x + block(aux)) / 2**0.5
            else:
                # x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                x = block(x)
                skips.append(x)

        new_skips = []
        #for x_tmp, conv_tmp in zip(skips, self.skip_conv_layer):
        #     x_tmp = conv_tmp(x_tmp)
        #     new_skips.append(x_tmp)

        #enumerates through the skip layers
        #self.skip_conv_layer is a list of our skip layers
        for idx, conv_tmp in enumerate(self.skip_conv_layer):
            #runs the data at the corresponding x layer through our skip layer
            x_tmp = conv_tmp(skips[idx])
            new_skips.append(x_tmp)

        #Decoder

        aux = None
        tmp = None

        #loops through all the decoder layers
        for name, block in self.dec.items():
            # print(name)
            # if "aux" not in name:
            #if this is true, then its because its a skip layer and we need to concatenate the skip layer with the current x layer
            if x.shape[1] != block.in_channels:
                # skip_ind = len(skips) - 1
                # skip_conv = self.skip_conv_layer[skip_ind]
                x = torch.cat([x, new_skips.pop()], dim=1)
            # x = block(x, emb)

            #we run our data through our block
            x = block(x)
            # else:
            #     # if "aux_up" in name:
            #     #     aux = block(aux)
            #     if "aux_conv" in name:
            #         tmp = block(silu(tmp))
            #         aux = tmp if aux is None else tmp + aux
            #     elif "aux_norm" in name:
            #         tmp = block(x)

        #runs our data through the normalization and convolutional layers at the very end
        for name, block in self.dec_aux_norm.items():
            tmp = block(x)

        #since we are running a standard decoder, there are no aux blocks so aux is None
        for name, block in self.dec_aux_conv.items():
            tmp = block(silu(tmp))
            aux = tmp if aux is None else tmp + aux

        # here x should be (batch, output_channels, seq_resolution)
        # remember that self.input_padding = (seq_resolution-self.vertical_level_num,0)
        x = aux
        # print('7:', x.shape)
        #extracts the transformed x_profile and x_scalar from x
        if self.input_padding[1]==0:
            y_profile = x[:,:self.target_profile_num,self.input_padding[0]:]
            y_scalar = x[:,self.target_profile_num:,self.input_padding[0]:]
        else:
            y_profile = x[:,:self.target_profile_num,self.input_padding[0]:-self.input_padding[1]]
            y_scalar = x[:,self.target_profile_num:,self.input_padding[0]:-self.input_padding[1]]

        #take relu on y_scalar
        y_scalar = torch.nn.functional.relu(y_scalar)
        #reshape y_profile to (batch, target_profile_num*levels)
        y_profile = y_profile.reshape(-1, self.target_profile_num*self.vertical_level_num)

        #average y_scalar for the lev dimension to (batch, target_scalar_num)
        #take the average scalar over the levels
        y_scalar = y_scalar.mean(dim=2)
        # print('7.5:', y_profile.shape, y_scalar.shape)

        #concatenate y_profile and y_scalar to (batch, target_profile_num*levels+target_scalar_num)
        y = torch.cat((y_profile, y_scalar), dim=1)

        #prunes the stratosphere values
        if self.output_prune:
            y = y.clone()
            y[:, 60:60+self.strato_lev_out] = y[:, 60:60+self.strato_lev_out].clone().zero_()
            y[:, 120:120+self.strato_lev_out] = y[:, 120:120+self.strato_lev_out].clone().zero_()
            y[:, 180:180+self.strato_lev_out] = y[:, 180:180+self.strato_lev_out].clone().zero_()
            y[:, 240:240+self.strato_lev_out] = y[:, 240:240+self.strato_lev_out].clone().zero_()
        return y