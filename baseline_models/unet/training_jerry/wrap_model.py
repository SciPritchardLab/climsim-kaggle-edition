import torch
import torch.nn as nn
import numpy as np

class WrappedModel(nn.Module):
    def __init__(self,
                 original_model,
                 input_sub,
                 input_div,
                 out_scale,
                 qn_lbd):
        super(WrappedModel, self).__init__()
        self.original_model = original_model
        self.input_sub = torch.tensor(input_sub, dtype=torch.float32, device = torch.device('cuda'))
        self.input_div = torch.tensor(input_div, dtype=torch.float32, device = torch.device('cuda'))
        self.out_scale = torch.tensor(out_scale, dtype=torch.float32, device = torch.device('cuda'))
        self.qn_lbd = torch.tensor(qn_lbd, dtype=torch.float32, device = torch.device('cuda'))

    def to(self, device):
        """Ensure all tensors are moved to the correct device"""
        self.input_sub = self.input_sub.to(device)
        self.input_div = self.input_div.to(device)
        self.out_scale = self.out_scale.to(device)
        self.qn_lbd = self.qn_lbd.to(device)
        return super().to(device)
    
    def apply_temperature_rules(self, T):
        # Create an output tensor, initialized to zero
        output = torch.zeros_like(T)

        # Apply the linear transition within the range 253.16 to 273.16
        mask = (T >= 253.16) & (T <= 273.16)
        output[mask] = (T[mask] - 253.16) / 20.0  # 20.0 is the range (273.16 - 253.16)

        # Values where T > 273.16 set to 1
        output[T > 273.16] = 1

        # Values where T < 253.16 are already set to 0 by the initialization
        return output

    def preprocessing(self, x):
        # convert v2 input array to v2_rh_mc input array:
        xout = x
        xout_new = torch.zeros((xout.shape[0], 557), dtype=xout.dtype, device = x.device)
        xout_new[:,0:120] = xout[:,0:120] # state_t, state_rh
        xout_new[:,120:180] = xout[:,120:180] + xout[:,180:240] # state_qn
        xout_new[:,180:240] = self.apply_temperature_rules(xout[:,0:60]) # liq_partition
        xout_new[:,240:557] = xout[:,240:557] # state_u, state_v
        x = xout_new
        
        #do input normalization
        x[:,120:180] = 1 - torch.exp(-x[:,120:180] * self.qn_lbd.to(x.device))
        x = (x - self.input_sub.to(x.device)) / self.input_div.to(x.device)
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[:,120:120+15] = 0
        #clip rh input
        x[:, 60:120] = torch.clamp(x[:, 60:120], 0, 1.2)

        return x

    def postprocessing(self, x):
        x[:,60:75] = 0
        x[:,120:135] = 0
        x[:,180:195] = 0
        x[:,240:255] = 0
        x = x/self.out_scale
        return x

    def forward(self, x):
        print(f"Model forward pass running on device: {x.device}")
        # Print the number of available CUDA devices
        num_gpus = torch.cuda.device_count()
        print(f"Number of available CUDA devices: {num_gpus}")
        t_before = x[:,0:60].clone()
        qc_before = x[:,120:180].clone()
        qi_before = x[:,180:240].clone()
        qn_before = qc_before + qi_before
        
        x = self.preprocessing(x)
        x = self.original_model(x)
        x = self.postprocessing(x)
        
        t_new = t_before + x[:,0:60]*1200.
        qn_new = qn_before + x[:,120:180]*1200.
        liq_frac = self.apply_temperature_rules(t_new)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        xout = torch.zeros((x.shape[0],368), device = x.device)
        xout[:,0:120] = x[:,0:120]
        xout[:,240:] = x[:,180:]
        xout[:,120:180] = (qc_new - qc_before)/1200.
        xout[:,180:240] = (qi_new - qi_before)/1200.
        return xout