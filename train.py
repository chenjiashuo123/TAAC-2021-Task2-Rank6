# coding: UTF-8
import time
import torch
import numpy as np
from src.util.train_efficient_fold_10_fgm import train_efficient_fgm
from src.util.train_vit_fold_10 import train_vit


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  


if __name__ == '__main__':
    train_vit()
    train_efficient_fgm()
    
    