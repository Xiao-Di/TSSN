import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Recurrent neural network (many-to-one)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # 初始化的隐藏元和记忆元,通常它们的维度是一样的
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) #x.size(0)是batch_size
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) # 2 for bidirectional

        # Forward propagate LSTM
        # x = x.permute(0,2,1)
        # print(x.shape) # [batch_size, seq_len, 1]
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print(out.shape) # [bs, seq_len, hidden*2]

        # Decode the hidden state of the last time step
        # out = F.softmax(self.fc(out[:, -1, :]))
        out = self.fc(out[:, -1, :])
        # print(out.shape) # [bs, 2]
        return out

