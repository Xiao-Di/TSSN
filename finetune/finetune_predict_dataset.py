from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random

class FinetunePredictDataset(Dataset):

    def __init__(self, file_path, seq_len, prediction_len,
                 num_features=1, word_len=6):
        # seq_len = 72, prediction_len = 12
        self.seq_len = seq_len
        self.prediction_len = prediction_len
        self.word_len = word_len
        self.num_words = int(seq_len / word_len)
        self.dimension = num_features
        # self.mode = mode

        df = pd.read_csv(file_path)
        self.Data = df.values
        print("Loading data from " + file_path + " successfully!")
        self.TS_num = self.Data.shape[0]

    def __len__(self):
        return self.TS_num

    def __getitem__(self, index):
        """
        mode='short' --> seq_len=72+12
        mode='medium' --> seq_len=72+36
        mode='long' --> seq_len=72+72
        """

        # 随机噪声
        # ts_masking, mask = self.random_masking(ts_processed, ts_length)

        ts_data = self.Data
        ### 要做归一化！！而且训练验证测试集上都要用训练集的最大最小值
        max = 369.78
        min = 0.00
        # if self.prediction_len == 1:
        #     min = 91.17
        #     max = 0.01
        # elif self.prediction_len == 6:
        #     min = 129.59
        #     max = 0.00
        # elif self.prediction_len == 12:
        #     min = 175.20
        #     max = 0.00
        # elif self.prediction_len == 18:
        #     min = 261.96
        #     max = 0.00
        # elif self.prediction_len == 24:
        #     min = 296.48
        #     max = 0.00
        # elif self.prediction_len == 30:
        #     min = 335.48
        #     max = 0.00
        # elif self.prediction_len == 36:
        #     min = 369.78
        #     max = 0.00
        # elif self.prediction_len == 42:
        #     min = 404.46
        #     max = 0.00
        # elif self.prediction_len == 48:
        #     min = 404.46
        #     max = 0.00
        # elif self.prediction_len == 54:
        #     min = 404.46
        #     max = 0.00
        # elif self.prediction_len == 60:
        #     min = 404.46
        #     max = 0.00
        # elif self.prediction_len == 66:
        #     min = 404.46
        #     max = 0.00
        # elif self.prediction_len == 72:
        #     min = 404.46
        #     max = 0.00
        # else:
        #     print("Not a legal prediction length...")
        # max = self.Data.max()
        # min = self.Data.min()

        ts_data_normalized = (ts_data - min) / (max - min)
        ts_array = np.array(ts_data_normalized[index])
        ts_masked = ts_array[:self.seq_len]
        bert_input = np.expand_dims(ts_masked, -1)  # (288,1)
        bert_input300 = np.expand_dims(ts_array[:self.seq_len + self.prediction_len], -1)

        # ts_masking = self.masking(ts_processed[:84])
        # bert_input = np.append(ts_processed[:72], np.array([100]*12))

        bert_mask = np.ones((self.num_words,), dtype=int)

        # bert_mask = np.expand_dims(bert_mask, -1)
        loss_mask = np.array([1] * self.prediction_len)
        # bert_target = ts_processed[:ts_length]
        bert_target = np.expand_dims(ts_array[self.seq_len:self.seq_len+self.prediction_len], -1)

        output = {
                  "bert_input": bert_input, # 时间序列 (288,1)
                  "bert_input300": bert_input300, # (300,1)
                  "bert_mask": bert_mask, # 有数据的地方是1,其他地方是0（全长seq_len） (48,)
                  "loss_mask": loss_mask, # 只计算要预测位置的loss (12,),预测的位置是1,其余位置是0
                  "bert_target": bert_target # (12,1)
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}


    # # 加入随机噪声
    # def masking(self, ts):

    #     ts_masking = ts.copy()
    #     mask = np.zeros((self.seq_len,), dtype=int)

    #     for i in range(self.seq_len//self.word_len):
    #         prob = random.random()
    #         # if prob < 0.15:
    #         #     prob /= 0.15
    #         #     mask[i] = 1

    #         if prob < 0.5:
    #             ts_masking[72+i, :] += np.random.uniform(low=-20, high=0, size=(self.dimension,))

    #         else:
    #             ts_masking[72+i, :] += np.random.uniform(low=0, high=20, size=(self.dimension,))

    #     return ts_masking