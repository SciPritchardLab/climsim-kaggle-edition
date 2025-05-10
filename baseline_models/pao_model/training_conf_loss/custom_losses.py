import torch
import torch.nn as nn

class MSELoss_conf(nn.Module):
    def __init__(self, conf_weight=1.0):
        super().__init__()
        self.MSE_no_reduce = nn.MSELoss(reduction='none')
        self.conf_weight = conf_weight
    
    def forward(self, preds, conf, target):
        loss = self.MSE_no_reduce(preds, target)
        loss2 = self.conf_weight * self.MSE_no_reduce(conf, loss.detach())
        combined_loss = loss + loss2
        return torch.mean(combined_loss)

class L1Loss_conf(nn.Module):
    def __init__(self, conf_weight=1.0):
        super().__init__()
        self.L1_no_reduce = nn.L1Loss(reduction='none')
        self.conf_weight = conf_weight
    
    def forward(self, preds, conf, target):
        loss = self.L1_no_reduce(preds, target)
        loss2 = self.conf_weight * self.L1_no_reduce(conf, loss.detach())
        combined_loss = loss + loss2
        return torch.mean(combined_loss)

class HuberLoss_conf(nn.Module):
    def __init__(self, conf_weight=1.0):
        super().__init__()
        self.Huber_no_reduce = nn.HuberLoss(reduction='none')
        self.conf_weight = conf_weight
    
    def forward(self, preds, conf, target):
        loss = self.Huber_no_reduce(preds, target)
        loss2 = self.conf_weight * self.Huber_no_reduce(conf, loss.detach())
        combined_loss = loss + loss2
        return torch.mean(combined_loss)