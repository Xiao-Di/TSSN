# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
# from torchsummary import summary
import sys
sys.path.append("..")
import os
from model.tvts_bert import TVTSBERT
from finetune.finetune_complement import TVTSBERTFTComplementator
from finetune_complement_dataset import FinetuneComplementDataset
import numpy as np
import random
from matplotlib import pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

word_len = 24
mask_len = 12
dropout = 0.1
file_path = '../data/'

pretrain_path = f'../checkpoints/pretrain/word-{word_len}-{dropout}/'
finetune_path = f'../checkpoints/finetune/complementation/1126/word-{word_len}-mask-{mask_len}-0/'

seq_len = 288
num_features = 1

pe_window = 288

epochs = 100
batch_size = 1
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5

train_file = file_path + 'train_complementation_smoothed.csv'
valid_file = file_path + 'valid_complementation_smoothed.csv'
test_file = file_path + 'com_test.csv'

print("Loading data sets...")
train_dataset = FinetuneComplementDataset(num_features=num_features,
                                          file_path=train_file,
                                          word_len=word_len,
                                          mask_len=mask_len,
                                          seq_len=seq_len)
valid_dataset = FinetuneComplementDataset(num_features=num_features,
                                          file_path=valid_file,
                                          word_len=word_len,
                                          mask_len=mask_len,
                                          seq_len=seq_len)
test_dataset = FinetuneComplementDataset(num_features=num_features,
                                         file_path=test_file,
                                         word_len=word_len,
                                         mask_len=mask_len,
                                         seq_len=seq_len)

print("Creating dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)

print("Initializing TVTS-BERT...")
tvtsbert = TVTSBERT(word_len=word_len,
                    pe_window=pe_window,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout)

# summary(tvtsbert, (64,1,144,144), (64,1,143,144))

print("Loading pretrained model parameters...")
tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))
# tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

print("Creating downstream complementation task FineTuner...")
finetuner = TVTSBERTFTComplementator(tvtsbert, num_features=num_features,
                                     seq_len=seq_len, word_len=word_len, mask_len=mask_len,
                                     train_dataloader=train_dataloader,
                                     valid_dataloader=valid_dataloader)


# print("Finetuning TVTS-BERT for Complementation word_len=%d mask_len=%d..." % (word_len, mask_len))
# for epoch in range(epochs):
#     train_loss, valid_loss = finetuner.train(epoch)
#     finetuner.save(epoch, finetune_path)


# Test: 重新加载finetune的模型
print("Testing TVTS-BERT Finetune Complementator...")
finetuner.load(finetune_path)

_ = finetuner.test(test_dataloader)
