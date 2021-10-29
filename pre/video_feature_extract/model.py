import sys
import torch as th
import torchvision.models as models
from torch import nn
import torch
import timm
from efficientnet_pytorch import EfficientNet


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = EfficientNet.from_name('efficientnet-b7')
        model = model.cuda()
        model_data = th.load('pretrain_models/efficient-b7/efficientnet-b7-dcc49843.pth')
        model.load_state_dict(model_data)
    model.eval()
    print('loaded')
    return model
