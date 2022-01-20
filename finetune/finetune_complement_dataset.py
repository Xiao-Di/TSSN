# -*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import random
import copy


class FinetuneComplementDataset(Dataset):
    def __init__(self,
                 # file_path='/Users/gengyunxin/Documents/项目/traffic_model/test/data/processed_pretrain_pems_72.csv',
                 num_features, file_path, word_len,
                 mask_len=12, seq_len=288):
        self.seq_len = seq_len
        self.word_len = word_len
        self.mask_len = mask_len
        self.num_words = seq_len // word_len
        self.num_masked_words = mask_len // word_len
        self.dimension = num_features

        df = pd.read_csv(file_path)
        self.Data = df.values # (591060, 72)
        print("Loading data successfully!")
        self.TS_num = self.Data.shape[0]

    def __len__(self):
        return self.TS_num

    def __getitem__(self, index):
        # ts_data = self.Data[:, 0:-1] # 不含label的数组 (4880,72)
        ts_data = self.Data

        # Normalize
        max = 638.3
        min = 0.0
        
        ts_data_normalized = (ts_data - min) / (max - min)
        # ts_data_normalized = ts_data # 不做归一化

        ts_processed = np.expand_dims(np.array(ts_data_normalized[index]), -1) # (288,1)

        ts_length = ts_processed.shape[0] # 288

        bert_mask = np.ones((self.num_words,), dtype=int)
        # bert_mask[:ts_length] = 1 # 其实seq_len和ts_length长度相同，都是72,全1数组。这么写是为了应对长度不等的情况 (72,)

        # 随机噪声
        ts_masking, origin12, loss_mask, mask288 = self.random_masking(ts_processed)
        # mask_ = np.expand_dims(mask, -1) # [12,1]

        # ts_target48 = np.zeros((self.seq_len,), dtype=float)
        # for index, item in enumerate(mask288):
        #     if item != 0:
        #         ts_target48[index] = ts_processed[index]
        # ts_target48_ = np.expand_dims(ts_target48, -1) # [288,1] 其他位置是0

        output = {"bert_input": ts_masking, # 加噪声的时间序列 (288,1)
                #   "bert_target48": ts_target48_, # (288,1) 除了48点mask外其他位置是0
                  "bert_target48": origin12, # [12,1] 加噪位置的真实值
                  "bert_target": ts_processed, # (288,1)
                  "bert_mask": bert_mask, # 有数据的地方是1（长度ts_length）,其他地方是0（全长seq_len） (num_words,)
                #   "loss_mask": mask288, # 只计算加噪声位置的loss (288,),加噪声的位置是1,其余位置是0
                  "loss_mask": loss_mask, # (12,) 全1
                  "position": mask288, # (288,1) 加噪位置是1其余位置是0
                  }
        # print(output)

        return {key: torch.from_numpy(value) for key, value in output.items()}


    def random_masking(self, ts):
        ts_masking = ts.copy()
        # ts_masking = copy.deepcopy(ts)
        mask288 = np.zeros((self.seq_len,), dtype=int)
        origin12 = np.zeros((self.mask_len,), dtype=int)
        mask48 = np.ones((self.mask_len,), dtype=int)

        a = self.num_words - self.num_masked_words # 每个seq一共有num_words-num_masked_words+1种被mask的可能
        # i = np.random.choice(a) # i是mask的 [词] 的起始位置
        index1 = 48
        index2 = 120
        # index3 = 168
        index4 = 216

        mask288[index4:index4 + self.mask_len] = 1
        origin12 = copy.deepcopy(ts_masking[index4:index4 + self.mask_len])
        ts_masking[index4:index4 + self.mask_len] = 0.0
        return ts_masking, origin12, mask48, mask288
