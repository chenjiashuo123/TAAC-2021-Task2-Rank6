import pandas as pd
import csv
import os
import json

from tqdm import tqdm
import time
from datetime import timedelta
from src.model.pytorch_pretrained import BertModel, BertTokenizer

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

Tokenizer = BertTokenizer.from_pretrained('pretrain_models/bert-base')
pad_size = 400

tag_dict = {}

with open('src/data/label_id.txt', 'r') as f:
    for line in f:
      line = line.strip().split('\t')
      tag_dict[line[0]] = line[1]
    print(tag_dict)


if os.path.exists("data_path.csv"):
    os.remove("data_path.csv")


cnt = 0

with open("src/data/Fold_data/data_path_all_vit.csv","w") as csvfile:
    contents = []
    for DIR_ in ['src/data/train.txt', 'src/data/val.txt']:
#     for DIR_ in ['train.txt']:

        with open(DIR_) as f:
            writer = csv.writer(csvfile)
            
            line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_train_5k', 'train').replace('Youtube8M/tagging', 'vit')
            while line:
                line_data = []
                # video
                line_data.append(line)
                # audio
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_train_5k', 'train').replace('Vggish/tagging/', '')
                line_data.append(line)
                # image
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_train_5k', 'train').replace('tagging/', '')
                line_data.append(line)
                
                # text
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_train_5k', 'train').replace('tagging/', '')
                with open(line,"r",encoding="utf-8") as f_:
                    data = json.load(f_)
                    text_ocr = data['video_ocr']
                    text_asr = data['video_asr']
                    line = [text_ocr, text_asr]
                    token_ids_twins = []
                    
                    for text in line:
                        token = Tokenizer.tokenize(text)
                        token = [CLS] + token
                        seq_len = len(token)
                        mask = []
                        token_ids = Tokenizer.convert_tokens_to_ids(token)

                        
                        if pad_size:
                            if len(token) < pad_size:
                                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                                token_ids += ([0] * (pad_size - len(token)))
                            else:
                                mask = [1] * pad_size
                                token_ids = token_ids[:pad_size]
                                seq_len = pad_size
                                
                        token_ids_twins.append((token_ids, seq_len, mask))
                    
                    
                line_data.append(token_ids_twins)
                
                # label
                line = f.readline().rstrip('\n').replace('..', '.')
                line_label = []
                for key_ in line.split(','):
                    line_label.append(tag_dict[key_])
                line_data.append(line_label)
                
                # \n
                line = f.readline()
                
                # next video
                contents.append(line_data)
                cnt += 1
                if cnt % 500 == 0:
                    print('Work Well! Processing in %d files.' % (cnt))
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_train_5k', 'train').replace('Youtube8M/tagging', 'vit')
                
    writer.writerow(["video_path","audio_path","image_path",'text_path','label'])
    writer.writerows(contents)



