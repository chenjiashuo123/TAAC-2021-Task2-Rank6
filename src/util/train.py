# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse
from utils import *


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
 
from models.fusion_net import Multi_Fusion_Net
from sklearn import metrics


from muli_dataset import MuliDataset
from torch.utils.data import DataLoader
import pandas as pd

from gap_cal import calculate_gap

from pytorch_pretrained.optimization import BertAdam

from sklearn.metrics import f1_score
from gap_cal import calculate_gap
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  


df_train = pd.read_csv('data_path_train.csv')
df_test = pd.read_csv('data_path_val.csv')


train_data = MuliDataset(df_train, device)
dev_data = MuliDataset(df_test, device)
test_data = MuliDataset(df_test, device)


lr = 1e-04
max_epoch = 20
batch_size = 28
num = 16

def net_train(net, train_loader, dev_iter):
    start_time = time.time()
    net.train()
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'bn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=0.05,
                         t_total=len(train_loader) * max_epoch)
    
    total_batch = 0 
    dev_best_acc = 0.
    last_improve = 0 
    flag = False
    

    net.train()
    for epoch in range(max_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, max_epoch))
        for i, (text_, img_, video_, audio_, label_) in enumerate(train_loader):
            combine_out  = net(text_, img_, video_, audio_)
            net.zero_grad()
            loss = F.binary_cross_entropy(combine_out, label_)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                train_acc = calculate_gap(combine_out.data.cpu(), label_.data.cpu(), top_k=20)
                dev_acc, dev_loss = evaluate(net, dev_iter, True)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train GAP: {2:>6.2%},  Val Loss: {3:>5.2},  Val GAP: {4:>6.2%},  Time: {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                if dev_best_acc < dev_acc:
                    print('!!!!!! Best model save!!!!!!!!!!!!')
                    torch.save(net, './best.pth')
                    dev_best_acc = dev_acc
                net.train()
            total_batch += 1
    return net


def evaluate(net, data_loader, mode=False):
    net.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    
    with torch.no_grad():
        for text_, img_, video_, audio_, labels in data_loader:
            outputs_ = net(text_, img_, video_, audio_)


            com_loss = F.binary_cross_entropy(outputs_, labels)
            loss_total += com_loss
            labels = labels.data.cpu().numpy()
            predic = outputs_.data.cpu().numpy()
            labels_all.extend(labels)
            predict_all.extend(predic)

    gap = calculate_gap(np.array(predict_all), np.array(labels_all), top_k=20)
    loss_total /= len(data_loader)
    if mode:
        return gap, loss_total
    
    print('GAP ON WHOLE TEST-DATA:', gap)
    
    return gap


if __name__ == '__main__':
    print("Loading data...")
    train_loader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size,
                        shuffle=True)
    video_dim = 768
    audio_dim = 768
    text_dim = 768
    
    net = Multi_Fusion_Net(16,video_dim, audio_dim, text_dim).cuda()
    unfreeze_layers = ['layer.9','layer.10', 'layer.11', 'bert.pooler', 'out.']
    model_asr = getattr(net, 'model_asr')
    for name, param in model_asr.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    model_ocr = getattr(net, 'model_ocr')
    for name, param in model_ocr.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    for name, param in model_ocr.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    net = nn.DataParallel(net)                       
    net = net_train(net=net, train_loader=train_loader, dev_iter=dev_loader)
    