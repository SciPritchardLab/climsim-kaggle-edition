from climsim_utils.data_utils import *

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import modulus

from resLSTM import resLSTM

class WrappedModel(nn.Module):
    def __init__(self, model_0, model_1, model_2, model_3, model_4, model_5, input_sub, input_div, out_scale, lbd_qn):
        super(WrappedModel, self).__init__()
        self.model_0 = model_0 # heating tendency
        self.model_1 = model_1 # moistening tendency
        self.model_2 = model_2 # liquid + ice tendency
        self.model_3 = model_3 # U tendency
        self.model_4 = model_4 # V tendency
        self.model_5 = model_5 # scalars
        self.input_sub = torch.tensor(input_sub, dtype=torch.float32)
        self.input_div = torch.tensor(input_div, dtype=torch.float32)
        self.out_scale = torch.tensor(out_scale, dtype=torch.float32)
        self.lbd_qn = torch.tensor(lbd_qn, dtype=torch.float32)
        self.input_series_num = 23
        self.input_single_num = 19
        self.output_series_num = 5
        self.output_single_num = 8
    
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
        
        # convert v4 input array to v5 input array:
        xout = x
        xout_new = torch.zeros((xout.shape[0], 1405), dtype=xout.dtype)
        xout_new[:,0:120] = xout[:,0:120]
        xout_new[:,120:180] = xout[:,120:180] + xout[:,180:240]
        xout_new[:,180:240] = self.apply_temperature_rules(xout[:,0:60])
        xout_new[:,240:840] = xout[:,240:840] #60*14
        xout_new[:,840:900] = xout[:,840:900]+ xout[:,900:960] #dqc+dqi
        xout_new[:,900:1080] = xout[:,960:1140]
        xout_new[:,1080:1140] = xout[:,1140:1200]+ xout[:,1200:1260]
        xout_new[:,1140:1405] = xout[:,1260:1525]
        x = xout_new
        
        #do input normalization
        x[:,120:180] = 1 - torch.exp(-x[:,120:180] * self.lbd_qn)
        mask = torch.ones(x.shape[1], dtype = torch.bool)
        indices_to_exclude = [-1, -4, -5, -6, -7, -8]
        mask[indices_to_exclude] = False
        x = x[:, mask]
        x= (x - self.input_sub) / self.input_div
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[:,120:120+15] = 0
        #clip rh input
        x[:, 60:120] = torch.clamp(x[:, 60:120], 0, 1.2)

        x_seq_series = x[:,:self.input_series_num*60].reshape(x.shape[0],self.input_series_num,60).permute(0,2,1) # b, 60, 23
        x_seq_single = x[:,self.input_series_num*60:].unsqueeze(1).repeat(1,60,1) # b, 60, 19
        x_seq = torch.cat([x_seq_series, x_seq_single], dim = -1) # b, 60, 42

        return x_seq

    def postprocessing(self, x):
        x[:,60:75] = 0
        x[:,120:135] = 0
        x[:,180:195] = 0
        x[:,240:255] = 0
        x = x/self.out_scale
        return x

    def forward(self, x):
        t_before = x[:,0:60].clone()
        qc_before = x[:,120:180].clone()
        qi_before = x[:,180:240].clone()
        qn_before = qc_before + qi_before
        
        x = self.preprocessing(x)
        x_0 = self.model_0(x)[:,0:60]
        x_1 = self.model_1(x)[:,60:120]
        x_2 = self.model_2(x)[:,120:180]
        x_3 = self.model_3(x)[:,180:240]
        x_4 = self.model_4(x)[:,240:300]
        x_5 = self.model_5(x)[:,300:]
        x = torch.cat([x_0, x_1, x_2, x_3, x_4, x_5], dim = -1)
        x = self.postprocessing(x)
        
        t_new = t_before + x[:,0:60]*1200.
        qn_new = qn_before + x[:,120:180]*1200.
        liq_frac = self.apply_temperature_rules(t_new)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        xout = torch.zeros((x.shape[0],368))
        xout[:,0:120] = x[:,0:120]
        xout[:,240:] = x[:,180:]
        xout[:,120:180] = (qc_new - qc_before)/1200.
        xout[:,180:240] = (qi_new - qi_before)/1200.
    
        return xout

def save_wrapper(model_0_path, model_1_path, model_2_path, model_3_path, model_4_path, model_5_path, casename):
    # casename = 'v5_noclassifier_huber_1y_noaggressive'

    f_inp_sub     = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/inp_sub.txt'
    f_inp_div     = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/inp_div.txt'
    f_out_scale   = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/out_scale.txt'
    f_qn_lbd = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/qn_exp_lambda_large.txt'
    lbd_qn = np.loadtxt(f_qn_lbd, delimiter=',')
    input_sub = np.loadtxt(f_inp_sub, delimiter=',')
    input_div = np.loadtxt(f_inp_div, delimiter=',')
    out_scale = np.loadtxt(f_out_scale, delimiter=',')

    model_0 = modulus.Module.from_checkpoint(model_0_path).to('cpu')
    model_1 = modulus.Module.from_checkpoint(model_1_path).to('cpu')
    model_2 = modulus.Module.from_checkpoint(model_2_path).to('cpu')
    model_3 = modulus.Module.from_checkpoint(model_3_path).to('cpu')
    model_4 = modulus.Module.from_checkpoint(model_4_path).to('cpu')
    model_5 = modulus.Module.from_checkpoint(model_5_path).to('cpu')

    wrapped_model = WrappedModel(model_0, model_1, model_2, model_3, model_4, model_5, input_sub, input_div, out_scale, lbd_qn)

    WrappedModel.device = "cpu"
    device = torch.device("cpu")
    scripted_model = torch.jit.script(wrapped_model)
    scripted_model = scripted_model.eval()
    save_file_torch = os.path.join('/global/homes/j/jerrylin/finetune_models', f'{casename}.pt')
    scripted_model.save(save_file_torch)
    return None

save_wrapper('/global/homes/j/jerrylin/resLSTM_512_v6_finetune_0/ckpt/ckpt_epoch_45_metric_0.1194.mdlus', \
             '/global/homes/j/jerrylin/resLSTM_512_v6_finetune_1/ckpt/ckpt_epoch_37_metric_0.1406.mdlus', \
             '/global/homes/j/jerrylin/resLSTM_512_v6_finetune_2/model.mdlus', \
             '/global/homes/j/jerrylin/resLSTM_512_v6_finetune_3/ckpt/ckpt_epoch_25_metric_0.0843.mdlus', \
             '/global/homes/j/jerrylin/resLSTM_512_v6_finetune_4/model.mdlus', \
             '/global/homes/j/jerrylin/resLSTM_512_v6_finetune_5/ckpt/ckpt_epoch_59_metric_0.0080.mdlus', \
             'resLSTM_finetuned')

# f_torch_model = '/global/homes/j/jerrylin/scratch/hugging/E3SM-MMF_ne4/saved_models/climsim3_intercomparison/resLSTM_512_v6/ckpt/ckpt_epoch_47_metric_0.3598.mdlus'
# f_inp_sub     = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/inp_sub.txt'
# f_inp_div     = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/inp_div.txt'
# f_out_scale   = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/out_scale.txt'
# f_qn_lbd = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training/normalization/qn_exp_lambda_large.txt'
# lbd_qn = np.loadtxt(f_qn_lbd, delimiter=',')
# input_sub = np.loadtxt(f_inp_sub, delimiter=',')
# input_div = np.loadtxt(f_inp_div, delimiter=',')
# out_scale = np.loadtxt(f_out_scale, delimiter=',')
# model_inf = modulus.Module.from_checkpoint(f_torch_model).to('cpu')

# wrapped_model = WrappedModel(model_inf, input_sub, input_div, out_scale, lbd_qn)

# WrappedModel.device = "cpu"
# device = torch.device("cpu")
