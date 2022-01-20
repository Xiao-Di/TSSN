# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
# from torchsummary import summary
import sys
sys.path.append("..")
import os
from LSTM.FC_completion_model import FC
from LSTM.FC_completion import FCComplementator
from finetune.finetune_complement_dataset import FinetuneComplementDataset
import numpy as np
import random
import datetime
import logging
from matplotlib import pyplot as plt
from model.early_stop import EarlyStopping

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(20)
patience = 15
num_features = 1
file_path = 'data/'
seq_len = 288
epochs = 100
batch_size = 1
hidden_size = 64
learning_rate = 2e-5
dropout = 0.1
word_len = 1
n_hidden_1 = 128
n_hidden_2 = 50


for mask_len in [12, 36]:
    print(f'Word length: {word_len}')
    early_stopping = EarlyStopping(patience, verbose=True)

    # pretrain_path = f'checkpoints/pretrain/word-{word_len}-{dropout}/' # the storage path of the pretrained model
    finetune_path = f'checkpoints/finetune/complementation/FC-mask-{mask_len}—0103/' # the output directory where the finetuning checkpoints written
    if not os.path.exists(finetune_path):
        os.makedirs(finetune_path)
    train_file = file_path + 'train_complementation_smoothed.csv'
    valid_file = file_path + 'valid_complementation_smoothed.csv'
    # test_file = file_path + 'test_complementation_smoothed.csv'
    test_file = file_path + 'com_test.csv'

    # logger.info("Loading data sets...")
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

    # logger.info("Creating dataloader...")
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, pin_memory=True,
                                    batch_size=batch_size, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                    batch_size=batch_size, drop_last=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                    batch_size=batch_size, drop_last=False)

    # logger.info("Initializing TVTS-BERT...")
    fc = FC(in_dim=seq_len * num_features,
            n_hidden_1=n_hidden_1,
            n_hidden_2=n_hidden_2,
            out_dim=mask_len)

    # summary(tvtsbert, (64,1,144,144), (64,1,143,144))

    # logger.info("Loading pretrained model parameters...")
    # tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
    # tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))
    # # tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

    # logger.info("Creating downstream complementation task FineTuner...")
    finetuner = FCComplementator(fc, num_features=num_features,
                                 seq_len=seq_len, mask_len=mask_len,
                                 n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
                                 train_dataloader=train_dataloader,
                                 valid_dataloader=valid_dataloader,
                                 lr=learning_rate)



    # Test: 重新加载finetune的模型
    # logger.info("\n" * 3)
    # logger.info("Testing TVTS-BERT Finetune Complementator...")
    finetuner.load(finetune_path)
    _ = finetuner.test(test_dataloader)

