import torch
import time
import os
import json
import heapq
import numpy as np
import pandas as pd
from src.data.dataset_test import MuliDataset
from torch.utils.data import DataLoader
from src.model.fusion_Bi_trans_effcient import Multi_Fusion_Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10



if os.path.exists("out.json"):
    os.remove("out.json")


tag_dict = {}

with open('src/data/label_id.txt', 'r') as f:
    for line in f:
      line = line.strip().split('\t')
      tag_dict[line[0]] = line[1]
    print(tag_dict)


tags_for_res_index = {int(value):key for key, value in tag_dict.items()}

res_to_write = {}


def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.

    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.

    Returns:
      The normalized prediction.
    """
#     print(predictions.shape)
    denominator = np.max(predictions) - np.min(predictions)
    
    ret = (predictions - np.min(predictions)) / max(denominator, epsilon)
    return ret


def pre_func(model, out_file=''):
    result = []

    pad_size = 50
    
    txt_root_dir = '../dataset/test_2nd/text_txt/'
    
    df_train = pd.read_csv('src/data/Fold_data/data_path_test_2nd_400_efficient.csv')
    train_data = MuliDataset(df_train, device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    
    
    cnt = 0
    
    file_list = os.listdir(txt_root_dir)
    file_list.sort()
    
    for i, (file_name, text_, video_, audio_) in enumerate(train_loader):
        if (i+1) % 10 == 0:
            print('Works Well On %dth File' % ((i+1) * batch_size))
        out_ = model(text_, video_, audio_)
        for (iter_, file_path) in zip(out_, file_name):
            sub_out = _zero_one_normalize(iter_.cpu().detach().numpy()).tolist()
#             top_k_index = heapq.nlargest(20, range(len(sub_out)), sub_out.__getitem__)
            res_to_write[file_path] = {"result" : [{"labels":[tags_for_res_index[iter_] for iter_ in range(82)],"scores":["%.2f" % sub_out[iter_] for iter_ in range(82)]}]}

def predict_efficient():
    for i in range(10):
        model = Multi_Fusion_Net(16,1536, 768, 768).cuda()
        model.load_state_dict(torch.load('checkpoint/efficient_best_fgm_{}.pth'.format(i)))
        model.eval()
        pre_func(model)
        with open('post-processing/inter_json/efficient_out_json_{}.json'.format(i), 'w') as f:
            json.dump(res_to_write, f, ensure_ascii=False, indent = 4) 
            
if __name__ == '__main__':
    
    for i in range(10):
        model = torch.load('checkpoint/efficient_best_fgm{}.pth'.format(i)).cuda()
        pre_func(model)
        with open('post-processing/inter_json/efficient_out_json_{}.json'.format(i), 'w') as f:
            json.dump(res_to_write, f, ensure_ascii=False, indent = 4) 
