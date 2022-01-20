import torch
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
sys.path.append("..")

# from model.tvts_bert import TVTSBERT
# from finetune_predict import TVTSBERTFTPredictor
from LSTM_prediction_model import LSTM
from LSTM_predict import LSTMPredictor
from finetune.finetune_predict_dataset import FinetunePredictDataset
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


def test2_plot(prediction_result_list, input289_list,  prediction_len, seq_len=288):
    p = []
    q = []
    inputs = []
    for item in prediction_result_list:

        p.append(item.cpu().numpy())
    for item in input289_list:
        q.append(item.cpu().numpy())


    # 预测值
    prediction_result = np.array(p[:-1]).flatten() # 去掉最后一个可能长度不一致的样本 len:4608
    print(prediction_result.shape)

    input289 = np.array(q[:-1]).flatten()

    for i in range(len(input289)):
        if i % (seq_len+prediction_len) == 0:
            inputs.append(input289[i:i+prediction_len])
    # 输入真实值
    inputs_array = np.array(inputs).flatten() # len:4608
    print(inputs_array.shape)


    plt.figure()
    # 输入曲线3*288蓝色
    # plt.plot(inputs_array[seq_len:seq_len*2], c='b', label='Input1')
    # plt.plot(np.arange(seq_len,seq_len*2), inputs_array[seq_len*2:seq_len*3], c='b', label='Origin2', linestyle='--')
    # plt.plot(np.arange(seq_len*2,seq_len*3), inputs_array[seq_len*3:seq_len*4], c='b', label='Origin3')
    plt.plot(inputs_array[seq_len:seq_len*8], c='r', label='Input')

    # 预测曲线3*288红色
    # plt.plot(prediction_result[:seq_len], c='r', label='Predict1')
    # plt.plot(np.arange(seq_len, seq_len*2), prediction_result[seq_len:seq_len*2], c='r', label='Predict2', linestyle='--')
    # plt.plot(np.arange(seq_len*2,seq_len*3), prediction_result[seq_len*2:seq_len*3], c='r', label='Predict3')
    plt.plot(prediction_result[:seq_len*7], c='g', label='Predict')

    plt.vlines(287.5, 0, 70, colors='black', linestyles='--')
    plt.vlines(575.5, 0, 70, colors='black', linestyles='--')
    plt.vlines(863.5, 0, 70, colors='black', linestyles='--')
    plt.vlines(1151.5, 0, 70, colors='black', linestyles='--')
    plt.vlines(1439.5, 0, 70, colors='black', linestyles='--')
    plt.vlines(1727.5, 0, 70, colors='black', linestyles='--')
    plt.hlines(0, 0, 2015, colors='black', linestyles='--')
    plt.title('A Week Prediction Result')
    plt.legend(['Input', 'Predict'])
    # plt.savefig('420finetune_predict_1.png')
    plt.savefig('2_12.png')

    df = pd.DataFrame({'Input': inputs_array[seq_len:seq_len * 8],
                       'Predict': prediction_result[:seq_len * 7],
                       })
    df.to_csv('2_12.csv')


def save_results(prediction_result_list, prediction_target_list, word_len, prediction_len, dropout, seq_len=288):
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
    print(f'Saving predict test result into test_result_all/prediction_result_step36/excel/lstm_{prediction_len}_{dropout}_step36.csv ...')
    df_all.to_csv(f'../finetune/test_result_all/prediction_result_step36/lstm_{prediction_len}_{dropout}_step36.csv')
    df_all.to_excel(f'../finetune/test_result_all/prediction_result_step36/excel/lstm_{prediction_len}_{dropout}_step36.xlsx', index=False)
    # df_all.to_excel(f'../finetune/test_result_all/prediction_result_step36/excel/lstm_{prediction_len}_plot.xlsx', index=False)
    # df_all.to_csv(f'../finetune/test_result_all/prediction_result_step36/lstm_{prediction_len}_plot.csv')


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

word_len = 1
# prediction_len = 12
dropout = 0.1

# for prediction_len in [1, 6, 12, 36, 72]:
for prediction_len in [36, 42, 48, 54]:
    print(f'prediction length: {prediction_len}')
    # pretrain_path = f'../checkpoints/pretrain/word-{word_len}-{dropout}/'
    # finetune_path = f'../checkpoints/finetune/prediction/word-{word_len}-p-{prediction_len}-dropout{dropout}-new/'
    finetune_path = f'../checkpoints/finetune/prediction/lstm-p{prediction_len}-dropout{dropout}-step36/'

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

    lstm = LSTM(input_size=num_features,
                hidden_size=hidden_size,
                num_layers=layers,
                prediction_len=prediction_len,
                )

    finetuner = LSTMPredictor(lstm, seq_len=seq_len, prediction_len=prediction_len,
                              hidden=hidden_size, num_features=num_features, layers=layers, lr=learning_rate,
                              train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

    # Test: 重新加载finetune的模型
    print("Testing LSTM Predictor...")


    finetuner.load(finetune_path)

    # prediction_result_list, prediction_target_list = finetuner.test(test_dataloader)
    prediction_result_list, prediction_target_list, input289_list = finetuner.test(test_dataloader)


    # test2_plot(prediction_result_list, input289_list, prediction_len=prediction_len)
    save_results(prediction_result_list, prediction_target_list,
        word_len=word_len, prediction_len=prediction_len, dropout=dropout)
    print('----------\n')
    print("\n" * 5)

