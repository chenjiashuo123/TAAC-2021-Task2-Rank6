import torch
from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=64, multiply=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1)
        
        out_avg = x*y_avg

        
        out = out_avg 
        return out