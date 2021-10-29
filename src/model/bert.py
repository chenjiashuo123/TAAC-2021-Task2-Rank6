# coding: UTF-8
import torch
import torch.nn as nn
from src.model.pytorch_pretrained import BertModel, BertTokenizer


class Bert(nn.Module):

    def __init__(self, hs):
        super(Bert, self).__init__()
        print('bert-base')
        self.bert = BertModel.from_pretrained('pretrain_models/bert-base')
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         print('mask', mask.shape)
        sequence_output, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        return pooled, sequence_output
    
