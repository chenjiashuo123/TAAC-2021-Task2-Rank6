import torch
import numpy as np
from models.bert import Bert
from models.NextVLAD import NeXtVLAD
from models.STAM import STAM
from models.transformer import Transformer
from models.longformer import Longformer
from models.cait import cait_XXS36_224
from models.LMF_TWO import LMF
from models.MULTModel import MULTModel
from models.channel_attention import SELayer
from models.classifier_head import Classifier_head

import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样


class Multi_Fusion_Net(nn.Module):
    def __init__(self, num, video_dim, audio_dim, text_dim):
        super(Multi_Fusion_Net, self).__init__()
        print('Inital video_text model')
        self.video_dim = video_dim
        self.text_dim = text_dim
        
        self.cat_dim = video_dim + text_dim
        
        self.model_asr = Bert(768)
        self.model_ocr = Bert(768)
        
        self.video_fc = nn.Linear(1024, 768)
        self.video_head = Longformer(seq_len=120, dim=768, depth=3, heads=12, dim_head=256, mlp_dim=3072, attention_window=8, dropout=0.1, emb_dropout=0.1)

        
        self.attention_block = SELayer(self.cat_dim)

        self.video_out_fc = nn.Linear(768*2, 768)
        self.video_out_fc_1 = nn.Linear(768, 256)
        self.video_out_fc_2 = nn.Linear(256, 82)





    def forward(self, text_, img_, video_, audio_):
        outputs_asr = self.model_asr(text_[0])
        outputs_ocr = self.model_ocr(text_[1])

        outputs_text = (outputs_asr + outputs_ocr) / 2


        outputs_video = self.video_fc(video_)
        outputs_video = self.video_head(outputs_video)


        outputs_ = torch.cat((outputs_text, outputs_video), dim=1).unsqueeze(2)
        outputs_ = self.attention_block(outputs_).squeeze(2)
        
        combine_out_ = self.video_out_fc(outputs_)
        combine_out_ = self.video_out_fc_1(combine_out_)
        combine_out_ = self.video_out_fc_2(combine_out_)
        combine_out_ = torch.sigmoid(combine_out_)
        
        return combine_out_


