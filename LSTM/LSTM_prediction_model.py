import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Recurrent neural network (many-to-one)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, prediction_len): # input_size 其实就是num_features=1
        super(LSTM, self).__init__() # lstm输入[batch_size, seq_len, input_size]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_len = prediction_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, input_size)

    def forward(self, x):
        # 初始化的隐藏元和记忆元,通常它们的维度是一样的
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) #x.size(0)是batch_size
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) # 2 for bidirectional

        # Forward propagate LSTM
        # x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        # print('out1:', out.shape)
        batch_size, seq_len, hidden = out.shape

        # Decode the hidden state of the last few time step
        out = out[:, -self.prediction_len:, :]
        # print('out2:', out.shape)
        # (batch_size, prediction_len, hidden_size * 2) -> (batch_size x prediction_len, hidden_size * 2)
        out = self.fc(out.contiguous().view(-1, hidden)) # view(batch_size * self.prediction_len, self.hidden_size * 2)
        # print('out3:', out.shape)
        # fc: (batch_size x prediction_len, hidden_size * 2) -> (batch_size x prediction_len, num_features)
        return out.view(batch_size, self.prediction_len, -1)