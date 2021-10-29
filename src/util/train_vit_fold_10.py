# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse
from src.util.utils import get_time_dif


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
 
from src.model.fusion_Bi_trans_vit import Multi_Fusion_Net
from sklearn import metrics


from src.data.muli_dataset import MuliDataset
from torch.utils.data import DataLoader
import pandas as pd

from src.util.gap_cal import calculate_gap
from src.util.adv_train import PGD, FGM

from src.model.pytorch_pretrained.optimization import BertAdam

from sklearn.metrics import f1_score
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  


lr = 2e-05
max_epoch = 10
batch_size = 16
num = 16

def net_train(fold, net, train_loader, dev_iter):
    start_time = time.time()
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
        for i, (text_, video_, audio_, label_) in enumerate(train_loader):
            total = 0
            correct = 0
            combine_out  = net(text_, video_, audio_)
            loss = F.binary_cross_entropy(combine_out, label_)
            loss.backward()
            optimizer.step()
            net.zero_grad()
            if total_batch % 50 == 0:
                train_acc = calculate_gap(combine_out.data.cpu(), label_.data.cpu(), top_k=20)
                total += 82 * batch_size
                correct += ((combine_out > 0.5) == label_).sum().item()
                dev_acc, dev_loss, acc_val = evaluate(net, dev_iter, True)

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train GAP: {2:>6.2%}, Train ACC: {3:>6.2%},  Val Loss: {4:>5.2},  Val GAP: {5:>6.2%}, Val ACC: {6:>6.2%},  Time: {7}'
                print(msg.format(total_batch, loss.item(), train_acc, correct / total, dev_loss, dev_acc, acc_val, time_dif))
                if dev_best_acc < dev_acc:
                    print('!!!!!! Best model save!!!!!!!!!!!!')
                    torch.save(net, 'checkpoint/vit_best_{}.pth'.format(fold))
                    dev_best_acc = dev_acc
                net.train()
            total_batch += 1
    return net


def evaluate(net, data_loader, mode=False):
    net.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    
    total = 0
    correct = 0
    with torch.no_grad():
        for text_, video_, audio_, labels in data_loader:
            outputs = net(text_, video_, audio_)

            loss = F.binary_cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all.extend(labels)
            predict_all.extend(predic)
            total += 82 * batch_size
            correct += ((predic>0.5) == labels).sum().item()


    gap = calculate_gap(np.array(predict_all), np.array(labels_all), top_k=20)
    acc =  correct / total
    

    if mode:
        return gap, loss_total / len(data_loader), acc
    
    print('GAP ON WHOLE TEST-DATA:', gap)
    
    return gap

def train_vit():
    for i in range(10):
        print("Loading Fold data {}".format(i))
        df_train = pd.read_csv('src/data/Fold_data/vit_10/data_path_vit_train_{}.csv'.format(i))
        df_dev = pd.read_csv('src/data/Fold_data/vit_10/data_path_vit_val_{}.csv'.format(i))
        train_data = MuliDataset(df_train, device)
        dev_data = MuliDataset(df_dev, device)

        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=batch_size,
                                shuffle=True)
        video_dim = 1536
        audio_dim = 768
        text_dim = 768

        net = Multi_Fusion_Net(16,video_dim, audio_dim, text_dim).cuda()
        unfreeze_layers = ['layer.9','layer.10', 'layer.11', 'out.']
        model_asr = getattr(net, 'model_asr')
        for name, param in model_asr.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        for name, param in model_asr.named_parameters():
            if param.requires_grad:
                print(name, param.size())                   
        net = net_train(i,net=net, train_loader=train_loader, dev_iter=dev_loader)
    


if __name__ == '__main__':
    
    for i in range(10):
        print("Loading  Fold data {}".format(i))
        df_train = pd.read_csv('src/data/Fold_data/vit_10/data_path_vit_train_{}.csv'.format(i))
        df_dev = pd.read_csv('src/data/Fold_data/vit_10/data_path_vit_val_{}.csv'.format(i))
        train_data = MuliDataset(df_train, device)
        dev_data = MuliDataset(df_dev, device)

        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=batch_size,
                                shuffle=True)
        video_dim = 1536
        audio_dim = 768
        text_dim = 768

        net = Multi_Fusion_Net(16,video_dim, audio_dim, text_dim).cuda()
        unfreeze_layers = ['layer.9','layer.10', 'layer.11', 'out.']
        model_asr = getattr(net, 'model_asr')
        for name, param in model_asr.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        for name, param in model_asr.named_parameters():
            if param.requires_grad:
                print(name, param.size())                   
        net = net_train(i,net=net, train_loader=train_loader, dev_iter=dev_loader)
    