import torch
import time
import os
import json
import heapq
import numpy as np
import pandas as pd


video_dir = 'post-processing/inter_json/'
file_list = os.listdir(video_dir)
file_list.sort()
results = []
for name in file_list:
    if 'json' in name:
        out_dir = video_dir + name
        with open(out_dir) as f:
            res_ = json.load(f)
        results.append(res_)
    
    



    

result_top_k = {}

tag_dict = {}

with open('src/data/label_id.txt', 'r') as f:
    for line in f:
      line = line.strip().split('\t')
      tag_dict[line[0]] = line[1]
    print(tag_dict)
    

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



tags_for_res_index = {int(value):key for key, value in tag_dict.items()}
print(len(results))

for key_ in list(results[0].keys()):
    scores_ = 0
    for out in results:
        scores_ += np.array([float(prob_) for prob_ in out[key_]['result'][0]['scores']])
    
    scores_ = scores_ / len(results)
    
    sub_out = _zero_one_normalize(scores_).tolist()
    top_k_index = heapq.nlargest(20, range(len(sub_out)), sub_out.__getitem__)
    
    result_top_k[key_] = {"result" : [{"labels":[tags_for_res_index[iter_] for iter_ in top_k_index],"scores":["%.2f" % sub_out[iter_] for iter_ in top_k_index]}]}

with open('out.json', 'w') as f:
    json.dump(result_top_k, f, ensure_ascii=False, indent = 4) 






