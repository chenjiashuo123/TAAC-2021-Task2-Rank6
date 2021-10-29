import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from src.model.blocks import (BridgeConnection, LayerStack,
                          PositionwiseFeedForward, ResidualConnection, clone, Identity, FeatureEmbedder)
from src.model.multihead_attention import MultiheadedAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)

    def forward(self, x, src_mask):
        '''
        in:
            x: (B, S, d_model), src_mask: (B, 1, S)
        out:
            (B, S, d_model)
        '''
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward

        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)

        return x

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        # torch.cuda.FloatTensor torch.FloatTensor
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        # same as input
        return x


class BiModalEncoderLayer(nn.Module):

    def __init__(self, d_model_M1, d_model_M2, d_model, dout_p, H, d_ff_M1, d_ff_M2):
        super(BiModalEncoderLayer, self).__init__()
        if d_model is not None:
            d_model_M1 = d_model
            d_model_M2 = d_model
        
        
        self.self_att_M1 = MultiheadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.self_att_M2 = MultiheadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M1 = MultiheadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M2 = MultiheadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)
        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)
        
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x, masks):
        '''
        Inputs:
            x (M1, M2): (B, Sm, Dm)
            masks (M1, M2): (B, 1, Sm)
        Output:
            M1m2 (B, Sm1, Dm1), M2m1 (B, Sm2, Dm2),
        '''
        M1, M2 = x
        M1_mask, M2_mask = masks

        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)

        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)

        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)

        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)

        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        # both (B, Sm*, Dm*)
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        # 2. Multimodal Attention (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)

        # 3. Feed-forward (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[2](M1m2, sublayer_ff_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class Encoder(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)

    def forward(self, x, src_mask):
        '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
        '''
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class BiModalEncoder(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V, N):
        super(BiModalEncoder, self).__init__()
        self.emb_A = Identity()
        self.emb_V = Identity()
#         self.pos_enc_A = PositionalEncoder(768, dout_p)
#         self.pos_enc_V = PositionalEncoder(768, dout_p)

        layer_AV = BiModalEncoderLayer(d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V)
        self.encoder_AV = LayerStack(layer_AV, N)
        
        

    def forward(self, x, masks: dict):
        '''
        Input:
            x (A, V): (B, Sm, D)
            masks: {V_mask: (B, 1, Sv); A_mask: (B, 1, Sa)}
        Output:
            (Av, Va): (B, Sm1, Dm1)
        '''
        A, V = x

        b, n, _ = A.shape
        
        A = self.emb_A(A)
        V = self.emb_V(V)


#         A = self.pos_enc_A(A)
#         V = self.pos_enc_V(V)

        # M1m2 (B, Sm1, D), M2m1 (B, Sm2, D) <-
        Av, Va = self.encoder_AV((A, V), (masks['A_mask'], masks['V_mask']))

        return (Av, Va)


if __name__ == '__main__':
    d_model_audio = 128
    d_model_video = 1024
    d_model = 1024
    dout_p = 0.1
    H = 4
    N = 2
    d_ff_audio = 3072
    d_ff_video = 3072

    video = torch.randn(2, 50, 1024)  # (batch x frames x channels x height x width)
    audio = torch.randn(2, 50, 128)
    v_mask = torch.ones(2, 1, 51)
    a_mask = torch.ones(2, 1, 51)
    masks = {'A_mask':a_mask, 'V_mask':v_mask}
    encoder = BiModalEncoder(128, 1024, None, 0.1, 4, 3072, 3072, 3)

    out = encoder((audio, video), masks)
    out_audio = out[0].mean(dim = 1)
    out_video = out[1][:, 1:]
    print(out_audio.shape)
    print(out_video.shape)
    print(out[1].shape)
