# -*-coding:utf-8-*-
import torch.nn as nn
import torch
import math
from model.tvts_bert import TVTSBERT

# 层归一化 LayerNorm
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GELU(nn.Module):
    """BERT used Gelu instead of Relu"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TVTSBERTFTComplementation(nn.Module):
    """
    Complementation task: data complementation for 48 dots;
    loss: MSE
    """

    def __init__(self, tvtsbert: TVTSBERT, seq_len, word_len, mask_len):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.mask_prediction = MaskedTimeSeriesModel(self.tvtsbert.hidden, seq_len, word_len, mask_len)

    def forward(self, x, mask):
        # print('x in:', x.shape)
        # print('mask in:', mask.shape)

        x = self.tvtsbert(x, mask)
        # print('x out tvtsbert:', x.shape)
        return self.mask_prediction(x)

class MaskedTimeSeriesModel(nn.Module):

    def __init__(self, hidden, seq_len, word_len, mask_len):
        super().__init__()
        self.w1_1 = int(seq_len * hidden / word_len) # w1_1=768 when K=24, 18432 when K=1
        self.w2_1 = self.w1_2 = 216
        self.linear1 = nn.Linear(self.w1_1, self.w1_2)
        self.linear2 = nn.Linear(self.w2_1, mask_len)
        self.activation = GELU()
        self.dropout = nn.Dropout(0.1)
        self.norm = LayerNorm(self.w1_2)

    def forward(self, x):
        # x shape [bs, N, hidden]
        x = torch.flatten(x, start_dim=1) # [bs, N*hidden]
        x = self.norm(self.dropout(self.activation(self.linear1(x)))) # [bs, 216]
        return self.linear2(x).unsqueeze(-1) # [bs, mask_len, 1]
