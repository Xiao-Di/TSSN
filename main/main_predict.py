# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
# from torchsummary import summary
import sys
sys.path.append("..")
import os
from model.tvts_bert import TVTSBERT
from finetune.finetune_predict import TVTSBERTFTPredictor
from finetune.finetune_predict_dataset import FinetunePredictDataset
import numpy as np
import random
import datetime
import logging
from matplotlib import pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(123)

file_path = 'data/prediction_compare/'

seq_len = 288
prediction_len = 6
num_features = 1
word_len = 24

pe_window = 288

epochs = 30
batch_size = 64
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")

info_log_file = f"log/prediction/K_{word_len}_P_{prediction_len}_{time_str}_info.log"
debug_log_file = f"log/prediction/K_{word_len}_P_{prediction_len}_{time_str}_debug.log"

logger = logging.getLogger(__name__)
debug_fhandler = logging.FileHandler(debug_log_file)
info_fhandler = logging.FileHandler(info_log_file)
debug_fmt = ("TIME: %(asctime)s - FILE: %(filename)s - FUNC: %(funcName)s\n"
             "%(levelname)s - LINE %(lineno)d - MSG: %(message)s")
info_fmt = "%(asctime)s - %(levelname)s: %(message)s"
debug_fhandler.setFormatter(logging.Formatter(debug_fmt))
debug_fhandler.setLevel(logging.DEBUG)
info_fhandler.setLevel(logging.INFO)
info_fhandler.setFormatter(logging.Formatter(info_fmt))
logger.setLevel(logging.DEBUG)
logger.addHandler(debug_fhandler)
logger.addHandler(info_fhandler)

pretrain_path = f'checkpoints/pretrain/word-{word_len}/' # the storage path of the pretrained model
finetune_path = 'checkpoints/finetune/prediction/word-%d-p%d/' % (word_len, prediction_len) # the output directory where the finetuning checkpoints written
if not os.path.exists(finetune_path):
    os.makedirs(finetune_path)

train_file = file_path + 'train_%d_smoothed.csv' % (seq_len+prediction_len)
valid_file = file_path + 'valid_%d_smoothed.csv' % (seq_len+prediction_len)
test_file = file_path + 'test_%d_smoothed.csv' % (seq_len+prediction_len)
# train_file = file_path + f'train_step36_pre{prediction_len}_smoothed.csv'
# valid_file = file_path + f'valid_step36_pre{prediction_len}_smoothed.csv'
# test_file = file_path + f'test_step36_pre{prediction_len}_smoothed.csv'

logger.info("Loading data sets...")
train_dataset = FinetunePredictDataset(file_path=train_file,
                                num_features=num_features,
                                seq_len=seq_len,
                                prediction_len=prediction_len,
                                word_len=word_len)
valid_dataset = FinetunePredictDataset(file_path=valid_file,
                                num_features=num_features,
                                seq_len=seq_len,
                                prediction_len=prediction_len,
                                word_len=word_len)
test_dataset = FinetunePredictDataset(file_path=test_file,
                               num_features=num_features,
                               seq_len=seq_len,
                               prediction_len=prediction_len,
                               word_len=word_len)

logger.info("Creating dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)

logger.info("Initializing TVTS-BERT...")
tvtsbert = TVTSBERT(word_len=word_len,
                    pe_window=pe_window,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout)

# summary(tvtsbert, (64,1,144,144), (64,1,143,144))

logger.info("Loading pretrained model parameters...")
tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
tvtsbert.load_state_dict(torch.load(tvtsbert_path))
# tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

logger.info("Creating downstream prediction task FineTuner...")
finetuner = TVTSBERTFTPredictor(tvtsbert, num_features=num_features,
                                seq_len=seq_len, prediction_len=prediction_len,
                                word_len=word_len, train_dataloader=train_dataloader,
                                valid_dataloader=valid_dataloader)


logger.info("Finetuning TVTS-BERT for Prediction...")
for epoch in range(epochs):
    train_loss, valid_loss = finetuner.train(epoch)
    finetuner.save(epoch, finetune_path)


# Test: 重新加载finetune的模型
# print("\n" * 5)
# print("Testing TVTS-BERT Finetune Predictor...")
# finetuner.load(finetune_path)
# # test需要输入参数i:第几个epoch(<= 31)
# # 参数j:第几个样本(<= 32)
# prediction_result_list, prediction_target_list = finetuner.test(test_dataloader)
# # 画图
# fig_plot(prediction_result_list, prediction_target_list, batch=10, sample=30,
#              seq_len=72, prediction_len=12)
