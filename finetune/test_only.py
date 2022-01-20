# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
sys.path.append("..")

from model.tvts_bert import TVTSBERT
from finetune_predict import TVTSBERTFTPredictor
# from LSTM.LSTM_prediction_model import LSTM
# from LSTM.LSTM_predict import LSTMPredictor
from finetune_predict_dataset import FinetunePredictDataset
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

def save_results(prediction_result_list, prediction_target_list, word_len, prediction_len, dropout, seq_len=288):
    # print(f'prediction result list shape: [{len(prediction_result_list)}, {len(prediction_result_list[0])}, {len(prediction_result_list[0][0])}]')
    # print(f'input list shape: [{len(input289_list)}, {len(input289_list[0])}, {len(input289_list[0][0])}]')
    # [20, batchsize, prediction len]
    # [20, batchsize, 288 + prediction len]
    r = []
    t = []
    inputs = []
    for item in prediction_result_list:
        r.append(item.cpu().numpy())
    for item in prediction_target_list:
        t.append(item.cpu().numpy())

    r_array = np.array(r[:-1]) # [num of sample in dataloader-1, batchsize, prediction_len]
    t_array = np.array(t[:-1]) # [num of sample in dataloader-1, batchsize, prediction_len]
    # 预测值
    prediction_result = r_array.flatten()
    # input289 = q_array[:,:,-prediction_len:].flatten()
    prediction_target = t_array.flatten()
    print(prediction_result.shape)
    print(prediction_target.shape)

    df_all = pd.DataFrame({'Input': prediction_target,
                           'Predict': prediction_result,
                       })
    print(df_all.head(10))
    # print(f'Saving predict test result into /{word_len}_{prediction_len}_{dropout}_new.xlsx ...')
    print(f'Saving predict test result into test_result_all/prediction_result_step36/excel/{word_len}_{prediction_len}_step36.xlsx ...')
    # df_all.to_excel(f'test_result_all/prediction_result_new/excel/lstm_{prediction_len}_{dropout}_new.xlsx', index=False)

    df_all.to_excel(f'test_result_all/prediction_result_step36/excel/{word_len}_{prediction_len}_step36_0104.xlsx', index=False)
    df_all.to_csv(f'test_result_all/prediction_result_step36/{word_len}_{prediction_len}_step36_0104.csv')
    # df_all.to_excel(f'test_result_all/prediction_result_step36/excel/{word_len}_{prediction_len}_plot.xlsx', index=False)
    # df_all.to_csv(f'test_result_all/prediction_result_step36/{word_len}_{prediction_len}_plot.csv')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

file_path = '../data/prediction_compare_step36/'
seq_len = 288
pe_window = 288
num_features = 1
epochs = 30
batch_size = 64
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5

# word_len = 1
# prediction_len = 11
dropout = 0.1

for word_len in [24]:
    print(f'Word_len: {word_len}')
    for prediction_len in [1,6,12,18,24,30,36,42,48,54,60,66,72]:
        print(f'prediction length: {prediction_len}')
        pretrain_path = f'../checkpoints/pretrain/word-{word_len}-{dropout}/'
        finetune_path = f'../checkpoints/finetune/prediction/0104/word-{word_len}-p-{prediction_len}-step36-0104/'
        # finetune_path = f'../checkpoints/finetune/prediction/transformer-p{prediction_len}-step36/'

        train_file = file_path + f'train_step36_pre{prediction_len}_smoothed.csv'
        valid_file = file_path + f'valid_step36_pre{prediction_len}_smoothed.csv'
        test_file = file_path + f'test_step36_pre{prediction_len}_smoothed1.csv' # MENTION the test file name
        # test_file = file_path + f'test_plot_{prediction_len}.csv' # plot test set

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

        tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
        tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

        # tvtsbert.load_state_dict(torch.load(tvtsbert_path))


        finetuner = TVTSBERTFTPredictor(tvtsbert, num_features=num_features,
                                        seq_len=seq_len, prediction_len=prediction_len,
                                        word_len=word_len, train_dataloader=train_dataloader,
                                        valid_dataloader=valid_dataloader)

        # Test: 重新加载finetune的模型
        print("Testing TVTS-BERT Finetune Predictor...")


        finetuner.load(finetune_path)

        # prediction_result_list, prediction_target_list = finetuner.test(test_dataloader)
        prediction_result_list, prediction_target_list, input289_list = finetuner.test(test_dataloader)

        save_results(prediction_result_list, prediction_target_list,
            word_len=word_len, prediction_len=prediction_len, dropout=dropout)
        print('----------')
        print("\n" * 5)
