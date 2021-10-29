import torch
import numpy as np
from src.model.bert import Bert
from src.model.NextVLAD import NeXtVLAD
from src.model.STAM import STAM
from src.model.transformer import Transformer
from src.model.longformer import Longformer
from src.model.cait import cait_XXS36_224
from src.model.LMF_TWO import LMF
from src.model.MULTModel import MULTModel
from src.model.channel_attention import SELayer
from src.model.classifier_head import Classifier_head
from src.model.encoders import BiModalEncoder
from src.model.eca_layer import eca_layer

import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class Multi_Fusion_Net(nn.Module):
    def __init__(self, num, video_dim, audio_dim, text_dim):
        super(Multi_Fusion_Net, self).__init__()
        print('Inital Bi_trans_effcient asr model')
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        
        self.text_dim = text_dim
        
        self.model_asr = Bert(768)
        
        self.video_fc = nn.Linear(2560, self.video_dim)
        self.audio_fc = nn.Linear(128, self.audio_dim)
        
        self.va_encoder = BiModalEncoder(audio_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.vt_encoder = BiModalEncoder(text_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.at_encoder = BiModalEncoder(audio_dim, text_dim, None, 0.1, 12, 3072, 3072, 2)
        
        self.video_head = NeXtVLAD(dim=video_dim, num_clusters=128,lamb=2, groups=16, max_frames=120)
        
        self.audio_head = NeXtVLAD(dim=audio_dim, num_clusters=128,lamb=2, groups=16, max_frames=120)
        
        self.text_head = NeXtVLAD(dim=text_dim, num_clusters=128,lamb=2, groups=16, max_frames=400)

        

        
        self.video_att = SELayer(video_dim*16)
        self.audio_att = SELayer(audio_dim*16)
        self.text_att = SELayer(text_dim*16)

        self.video_out_fc = nn.Linear(self.video_dim*16, 82)
        self.audio_out_fc = nn.Linear(self.audio_dim*16, 82)
        self.text_out_fc = nn.Linear(self.text_dim*16, 82)
        





    def forward(self, text_, video_, audio_):
        outputs_video = video_[0]
        outputs_audio = audio_[0]
        text_asr = text_[1]
        
        pool_outputs_asr, sequence_output_asr = self.model_asr(text_asr)
        outputs_video = self.video_fc(outputs_video)
        outputs_audio = self.audio_fc(outputs_audio)

        
        va_masks = {'A_mask':audio_[1].unsqueeze(1), 'V_mask':video_[1].unsqueeze(1)}
        vt_masks = {'A_mask':text_asr[2].unsqueeze(1), 'V_mask':video_[1].unsqueeze(1)}
        at_masks = {'A_mask':audio_[1].unsqueeze(1), 'V_mask':text_asr[2].unsqueeze(1)}
        

        
        out_va = self.va_encoder((outputs_audio, outputs_video), va_masks)
        out_vt = self.vt_encoder((sequence_output_asr, outputs_video), vt_masks)
        out_at = self.at_encoder((outputs_audio, sequence_output_asr), at_masks)
        
        
        va_audio_out = out_va[0]
        va_video_out = out_va[1]
        
        vt_text_out = out_vt[0]
        vt_video_out = out_vt[1]
        
        at_audio_out = out_at[0]
        at_text_out = out_at[1]

        
        video_out = va_video_out + vt_video_out
        audio_out = va_audio_out + at_audio_out
        text_out = vt_text_out + at_text_out
        
        
        video_out = self.video_head(video_out).unsqueeze(2)
        audio_out = self.audio_head(audio_out).unsqueeze(2)
        text_out = self.text_head(text_out).unsqueeze(2)
       

        audio_out = self.audio_att(audio_out).squeeze(2)
        video_out = self.video_att(video_out).squeeze(2)
        text_out = self.text_att(text_out).squeeze(2)

        
    
        video_out = self.video_out_fc(video_out)
        video_out = torch.sigmoid(video_out)
        
        text_out = self.text_out_fc(text_out)
        text_out = torch.sigmoid(text_out)
        
        audio_out = self.audio_out_fc(audio_out)
        audio_out = torch.sigmoid(audio_out)

        combine_out_ = (video_out + text_out + audio_out) / 3
        return combine_out_



