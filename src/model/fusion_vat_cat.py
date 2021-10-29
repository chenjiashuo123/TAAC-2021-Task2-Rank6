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
from models.encoders import BiModalEncoder
from models.eca_layer import eca_layer

import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class Multi_Fusion_Net(nn.Module):
    def __init__(self, num, video_dim, audio_dim, text_dim):
        super(Multi_Fusion_Net, self).__init__()
        print('Inital video_text_audio_cat model')
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        
        self.cat_dim = video_dim*16 + audio_dim*16 + text_dim
        
        self.model_asr = Bert(768)
        
        self.video_head = NeXtVLAD(dim=1024, num_clusters=128,lamb=2, groups=16, max_frames=120)
        self.audio_head = NeXtVLAD(dim=128, num_clusters=128,lamb=2, groups=16, max_frames=120)


        self.attention_block = SELayer(self.cat_dim)
        self.video_out_fc_1 = nn.Linear(self.cat_dim, 82)





    def forward(self, text_, video_, audio_):
        
        outputs_video = video_[0]
        outputs_audio = audio_[0]
        text_asr = text_[1]
        
        pool_outputs_asr, sequence_output_asr = self.model_asr(text_asr)
        
        outputs_video = self.video_head(outputs_video)
#         outputs_video = self.video_fc(outputs_video)

        outputs_audio = self.audio_head(outputs_audio)
    

        outputs_ = torch.cat((pool_outputs_asr, outputs_video, outputs_audio), dim=1).unsqueeze(2)
        
        combine_out_ = self.attention_block(outputs_).squeeze(2)
        combine_out_ = self.video_out_fc_1(combine_out_)
        
        combine_out_ = torch.sigmoid(combine_out_)
        
        return combine_out_


