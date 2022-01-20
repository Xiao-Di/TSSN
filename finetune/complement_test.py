# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
sys.path.append("..")

from model.tvts_bert import TVTSBERT
from finetune_complement import TVTSBERTFTComplementator
from finetune_complement_dataset import FinetuneComplementDataset
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def test2_plot(complementation288_list, input288_list, seq_len=288):
    print("Start processing image...")
    p = []
    q = []
    for item in complementation288_list:
        p.append(item.cpu().numpy())
    for item in input288_list:
        q.append(item.cpu().numpy())

    # 预测值
    complementation_result = np.array(p[:-1]).flatten() # 去掉最后一个可能长度不一致的样本 (6488064,)

    # 输入真实值
    input288 = np.array(q[:-1]).flatten() # (6488064,)

    plt.figure()
    # 输入曲线288红色
    plt.plot(input288[:seq_len*3], c='r', label='Input')

    # 预测曲线288蓝色
    plt.plot(complementation_result[:seq_len*3], c='blue', label='Complement')

    # plt.vlines(287.5, 0, 70, colors='black', linestyles='--')
    # plt.vlines(575.5, 0, 70, colors='black', linestyles='--')
    # plt.vlines(863.5, 0, 70, colors='black', linestyles='--')
    # plt.vlines(1151.5, 0, 70, colors='black', linestyles='--')
    # plt.vlines(1439.5, 0, 70, colors='black', linestyles='--')
    # plt.vlines(1727.5, 0, 70, colors='black', linestyles='--')
    # plt.hlines(0, 0, 2015, colors='black', linestyles='--')
    plt.title('One Day Complementation Result')
    plt.legend(['Input', 'Complement'])
    # plt.savefig('420finetune_predict_1.png')
    plt.savefig('Complement12—6.png')
    print("PNG saved!")


def save_results(complementation288_list, input48_list, mask_len):
    print("Start processing CSV...")
    p = []
    q = []
    complementation48 = []
    input48 = []
    for item in complementation288_list:
        p.append(item.cpu().numpy())
    for item in input48_list:
        q.append(item.cpu().numpy())

    # 预测值
    complementation288 = np.array(p[:-1]).flatten()

    input48_array = np.array(q[:-1]).flatten()

    # for item in complementation288:
    #     if item != 0.0:
    #         complementation48.append(item)
    for ind, item in enumerate(input48_array):
        if item != 0.0:
            input48.append(item)
            complementation48.append(complementation288[ind])

    mse = mean_squared_error(input48[:mask_len], complementation48[:mask_len])
    print('mse:', mse)
    print(np.array(complementation48).shape)
    print(np.array(input48).shape)

    df_all = pd.DataFrame({'Input': np.array(input48),
                           'Predict': np.array(complementation48),
                       })
    df_all.to_csv('test_result_all/Complementation12-6.csv')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

file_path = '../data/'

pretrain_path = '../checkpoints/pretrain/word12/' # the storage path of the pretrained model
finetune_path = '../checkpoints/finetune/' # the output directory where the finetuning checkpoints written

seq_len = 288
pe_window = 288

mask_len = 24
num_features = 1
word_len = 12

epochs = 50
batch_size = 256
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

train_file = file_path + 'train_complementation_smoothed.csv'
valid_file = file_path + 'valid_complementation_smoothed.csv'
test_file = file_path + 'test_complementation_smoothed.csv'

print("Lodaing data sets...")
train_dataset = FinetuneComplementDataset(num_features=num_features,
                                          file_path=train_file,
                                          word_len=word_len,
                                          mask_len=mask_len,
                                          seq_len=seq_len)
valid_dataset = FinetuneComplementDataset(num_features=num_features,
                                          file_path=train_file,
                                          word_len=word_len,
                                          mask_len=mask_len,
                                          seq_len=seq_len)
test_dataset = FinetuneComplementDataset(num_features=num_features,
                                         file_path=train_file,
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

tvtsbert = TVTSBERT(word_len=word_len,
                    hidden=hidden_size,
                    n_layers=layers,
                    pe_window=pe_window,
                    attn_heads=attn_heads,
                    dropout=dropout)

print("Loading pretrained model parameters...")
tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
tvtsbert.load_state_dict(torch.load(tvtsbert_path))
# tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

print("Creating downstream complementation task FineTuner...")
finetuner = TVTSBERTFTComplementator(tvtsbert, num_features=num_features,
                                     seq_len=seq_len, word_len=word_len, mask_len=mask_len,
                                     train_dataloader=train_dataloader,
                                     valid_dataloader=valid_dataloader)



# Test: 重新加载finetune的模型
print("\n" * 3)
print("Testing TVTS-BERT Finetune Complementator...")

finetuner.load(finetune_path)

test_error = finetuner.test(test_dataloader)

# 画图
# test2_plot(complementation288_list, input288_list)
# save_results(complementation288_list, input288_list, mask_len=mask_len)

