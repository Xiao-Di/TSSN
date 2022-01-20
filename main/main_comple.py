# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
# from torchsummary import summary
import sys
sys.path.append("..")
import os
from model.tvts_bert import TVTSBERT
from finetune.finetune_complement import TVTSBERTFTComplementator
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


setup_seed(20)

patience = 10

num_features = 1
pe_window = 288

file_path = 'data/'
seq_len = 288
epochs = 100
batch_size = 64
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1


for mask_len in [36,12]:
    for word_len in [1,2,6,12,24]:
        print(f'Word length: {word_len}')
        early_stopping = EarlyStopping(patience, verbose=True)

        # time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        # info_log_file = f"log/complementation/K_{word_len}_mask_{mask_len}_dropout{dropout}_{time_str}_info.log"
        # debug_log_file = f"log/complementation/K_{word_len}_mask_{mask_len}_dropout{dropout}_{time_str}_debug.log"
        # logger = logging.getLogger(__name__)
        # debug_fhandler = logging.FileHandler(debug_log_file)
        # info_fhandler = logging.FileHandler(info_log_file)
        # debug_fmt = ("TIME: %(asctime)s - FILE: %(filename)s - FUNC: %(funcName)s\n"
        #             "%(levelname)s - LINE %(lineno)d - MSG: %(message)s")
        # info_fmt = "%(asctime)s - %(levelname)s: %(message)s"
        # debug_fhandler.setFormatter(logging.Formatter(debug_fmt))
        # debug_fhandler.setLevel(logging.DEBUG)
        # info_fhandler.setLevel(logging.INFO)
        # info_fhandler.setFormatter(logging.Formatter(info_fmt))
        # logger.setLevel(logging.DEBUG)
        # logger.addHandler(debug_fhandler)
        # logger.addHandler(info_fhandler)

        pretrain_path = f'checkpoints/pretrain/word-{word_len}-{dropout}/' # the storage path of the pretrained model
        finetune_path = f'checkpoints/finetune/complementation/1126/word-{word_len}-mask-{mask_len}-dropout{dropout}/' # the output directory where the finetuning checkpoints written
        if not os.path.exists(finetune_path):
            os.makedirs(finetune_path)
        train_file = file_path + 'train_complementation_smoothed.csv'
        valid_file = file_path + 'valid_complementation_smoothed.csv'
        test_file = file_path + 'test_complementation_smoothed.csv'

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
        tvtsbert = TVTSBERT(word_len=word_len,
                            pe_window=pe_window,
                            hidden=hidden_size,
                            n_layers=layers,
                            attn_heads=attn_heads,
                            dropout=dropout)

        # summary(tvtsbert, (64,1,144,144), (64,1,143,144))

        # logger.info("Loading pretrained model parameters...")
        tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
        tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))
        # tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

        # logger.info("Creating downstream complementation task FineTuner...")
        finetuner = TVTSBERTFTComplementator(tvtsbert, num_features=num_features,
                                            seq_len=seq_len, word_len=word_len, mask_len=mask_len,
                                            train_dataloader=train_dataloader,
                                            valid_dataloader=valid_dataloader)


        # logger.info(f"Finetuning TVTS-BERT for Complementation K={word_len} mask={mask_len} dropout={dropout}...")
        # for epoch in range(epochs):
        #     train_loss, valid_loss = finetuner.train(epoch)
        #     early_stopping(valid_loss, finetuner)
        #     # finetuner.save(epoch, finetune_path)
        #     print(f'Early stopping counter: {early_stopping.counter}')
        #     if  early_stopping.counter == 0:
        #         finetuner.save(epoch, finetune_path)
        #     if early_stopping.early_stop:
        #         print(f'Early stopping at {epoch}...')
        #         break

        # Test: 重新加载finetune的模型
        # logger.info("\n" * 3)
        # logger.info("Testing TVTS-BERT Finetune Complementator...")
        finetuner.load(finetune_path)

        _ = finetuner.test(test_dataloader)

