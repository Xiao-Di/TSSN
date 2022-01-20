import torch
from pretrain.pretrain_dataset import DatasetWrapper
import sys
sys.path.append("..")

from model.tvts_bert import TVTSBERT
from pretrain.pretrain import TVTSBERTTrainer
import numpy as np
import random
import datetime

import logging

# 5 NOTEs to notice

# 每次预训练之前，改一下pretraining.py里面writter的地址
# 还要改一下模型保存地址checkpoints
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(123)

valid_rate = 0.2
max_length = 288
word_len = 24 # note2: 改这个参数
pe_window = 288
num_features = 1
epochs = 50
batch_size = 128 # note3: batch size可以加
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 1e-4
warmup_epochs = 3
decay_gamma = 0.99
dropout = 0.1
gradient_clipping = 5.0

time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")

info_log_file = f"log/K_{word_len}_{time_str}_info.log"
debug_log_file = f"log/K_{word_len}_{time_str}_debug.log"

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

dataset_path = 'data/processed_pretrain_pems_288.csv'
# dataset_path = "data/test.csv"
# dataset_path = '/panfs/sugon/gpu/home/caol1/Pywork/gyx/traffic_model/giteee/data/processed_pretrain_pems_288.csv'
pretrain_path = f'checkpoints/pretrain/word-{word_len}-{time_str}/'  # note4: checkpoint保存的地址

logger.info("Loading training and validation data sets...")
dataset = DatasetWrapper(batch_size=batch_size,
                         valid_ratio=valid_rate,
                         data_path=dataset_path,
                         num_features=num_features,
                         max_length=max_length,
                         word_len=word_len)

# training set split
train_loader, valid_loader = dataset.get_data_loaders()

logger.info("Initialing TVTS-BERT...")
tvtsbert = TVTSBERT(word_len=word_len,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout,
                    pe_window=pe_window)

trainer = TVTSBERTTrainer(tvtsbert, num_features=num_features,
                          word_len=word_len,
                          train_dataloader=train_loader,
                          valid_dataloader=valid_loader,
                          lr=learning_rate,
                          warmup_epochs=warmup_epochs,
                          decay_gamma=decay_gamma,
                          gradient_clipping_value=gradient_clipping)

logger.info("Pretraining TVTS-BERT...")
mini_loss = np.Inf
for epoch in range(epochs):
    train_loss, valid_loss = trainer.train(epoch)
    logger.info(f"Training round: {epoch}/{epochs}, current loss: {train_loss}")
    if mini_loss > valid_loss:
        mini_loss = valid_loss
        trainer.save(epoch, pretrain_path)
