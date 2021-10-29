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

cnt = 0

with open("src/data/Fold_data/data_path_test_2nd_400_vit.csv","w") as csvfile:
    contents = []
    for DIR_ in ['src/data/test_2nd.txt']:
#     for DIR_ in ['train.txt']:

        with open(DIR_) as f:
            writer = csv.writer(csvfile)
            
            line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_test_5k_2nd', 'test_2nd').replace('Youtube8M/tagging', 'vit')
            while line:
                line_data = []
                # video
                line_data.append(line)
                # audio
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_test_5k_2nd', 'test_2nd').replace('Vggish/tagging/', '')
                line_data.append(line)
                # image
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_test_5k_2nd', 'test_2nd').replace('tagging/', '')
                line_data.append(line)
                
                # text
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_test_5k_2nd', 'test_2nd').replace('tagging/', '')
                with open(line,"r",encoding="utf-8") as f_:
                    data = json.load(f_)
                    text_ocr = data['video_ocr']
                    text_asr = data['video_asr']
                    if text_ocr == '':
                        text_ocr = text_asr
                    if text_asr == '':
                        text_asr = text_ocr
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
                
                
                # \n
                line = f.readline()
                
                # next video
                contents.append(line_data)
                cnt += 1
                if cnt % 500 == 0:
                    print('Work Well! Processing in %d files.' % (cnt))
                line = f.readline().rstrip('\n').replace('tagging/tagging_dataset_test_5k_2nd', 'test_2nd').replace('Youtube8M/tagging', 'vit')
                
    writer.writerow(["video_path","audio_path","image_path",'text_path'])
    writer.writerows(contents)



