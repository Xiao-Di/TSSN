import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append("..")
from finetune.finetune_dataset import FinetuneDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FC(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
