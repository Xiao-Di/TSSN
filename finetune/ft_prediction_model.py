import torch.nn as nn
import torch
import math
from model.tvts_bert import TVTSBERT

class GELU(nn.Module):
    """BERT used Gelu instead of Relu"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TVTSBERTFTPrediction(nn.Module):
    """
    Prediction task:predict contaminated data;
    loss: MSE
    """

    def __init__(self, tvtsbert: TVTSBERT, num_features, seq_len, prediction_len, word_len):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.prediction = PredictionModel(self.tvtsbert.hidden, num_features,
                                          seq_len, prediction_len, word_len)

    def forward(self, x, mask):
        # print('x in:', x.shape)
        # print('mask in:', mask.shape)

        x = self.tvtsbert(x, mask)
        # print('x out tvtsbert:', x.shape)
        return self.prediction(x)


class PredictionModel(nn.Module):

    def __init__(self, hidden, num_features, seq_len, prediction_len, word_len):
        super().__init__()
        self.num_words = int(seq_len / word_len)
        self.linear1 = nn.Linear(hidden, num_features) # in_features输入二维张量大小, out_features输出二维张量大小
        self.linear2 = nn.Linear(self.num_words, prediction_len) # 输入数据长度、预测数据长度
        self.activation = GELU()

    def forward(self, x):
        # x [batchsize, num_words, hidden_size]
        # print('x out tvts bert:', x.shape)
        x = self.activation(self.linear1(x)).squeeze() # [batchsize, num_words]
        # print('x out linear1: ', x.shape)
        # print('x out linear2:', self.linear2(x).unsqueeze(-1).shape)
        # print('size after linear1 and squeeze: ', x.shape)
        return self.linear2(x).unsqueeze(-1)