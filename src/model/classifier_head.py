import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


class Classifier_head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier_head, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.out_fc_1 = nn.Linear(self.in_dim, self.in_dim//2)
        self.out_fc_2 = nn.Linear(self.in_dim//2, self.in_dim//4)
        self.out_fc_3 = nn.Linear(self.in_dim//4, 82)
        
    def forward(self, input):
        outputs_last = F.relu(self.out_fc_1(input))
        outputs_last = self.out_fc_2(outputs_last)
        outputs_last = self.out_fc_3(outputs_last)
        outputs_ = torch.sigmoid(outputs_last)
        
        return outputs_