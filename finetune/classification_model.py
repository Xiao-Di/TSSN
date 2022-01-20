import torch
import torch.nn as nn
from model.tvts_bert import TVTSBERT

class TVTSBERTClassification(nn.Module):
    """
    Downstream finetune task: weekday or weekend classification
    """

    def __init__(self, tvtsbert: TVTSBERT, num_classes, seq_len, word_len):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.classification = MulticlassClassification(self.tvtsbert.hidden, num_classes, seq_len, word_len)

    def forward(self, x, mask): # data['bert_input'], data['bert_mask']
        x = self.tvtsbert(x, mask)
        return self.classification(x)



class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes, seq_len, word_len):
        super().__init__()
        self.pooling_len = int(seq_len / word_len)
        self.pooling = nn.MaxPool1d(self.pooling_len)  # 可以换成avgpooling对比看看效果
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # print(x.shape) # [128, 36, 64]
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.pooling(x)
        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        x = self.linear(x)
        return x