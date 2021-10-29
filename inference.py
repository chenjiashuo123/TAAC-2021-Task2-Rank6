# coding: UTF-8
import time
import torch
import numpy as np
from src.util.predict_fold_efficient import predict_efficient
from src.util.predict_fold_vit import predict_vit




if __name__ == '__main__':
    predict_vit()
    predict_efficient()