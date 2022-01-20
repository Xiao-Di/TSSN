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


def save_results(prediction_result_list, input289_list, word_len, prediction_len, dropout, seq_len=288):
    # print(f'prediction result list shape: [{len(prediction_result_list)}, {len(prediction_result_list[0])}, {len(prediction_result_list[0][0])}]')
    # print(f'input list shape: [{len(input289_list)}, {len(input289_list[0])}, {len(input289_list[0][0])}]')
    # [20, batchsize, prediction len]
    # [20, batchsize, 288 + prediction len]
    p = []
    q = []
    inputs = []
    for item in prediction_result_list:
        p.append(item.cpu().numpy())
    for item in input289_list:
        q.append(item.cpu().numpy())

    p_array = np.array(p[:-1]) # [num of sample in dataloader-1, batchsize, prediction_len]
    q_array = np.array(q[:-1]) # [num of sample in dataloader-1, batchsize, 288 + prediction_len]
    # 预测值
    prediction_result = p_array.flatten()
    input289 = q_array[:,:,-prediction_len:].flatten()
    print(prediction_result.shape)
    print(input289.shape)

    df_all = pd.DataFrame({'Input': input289,
                           'Predict': prediction_result,
                       })
    print(df_all.head(10))
    # df.to_csv('420predict_result_1.csv')
    # print(f'Saving predict test result into /{word_len}_{prediction_len}_{dropout}_new.xlsx ...')
    print(f'Saving predict test result into test_result_all/prediction_result_new/excel/lstm_{prediction_len}_{dropout}_step36.csv ...')
    df_all.to_csv(f'../finetune/test_result_all/prediction_result_new/lstm_{prediction_len}_{dropout}_step36.csv')
    df_all.to_excel(f'../finetune/test_result_all/prediction_result_new/excel/lstm_{prediction_len}_{dropout}_step36.xlsx', index=False)
    # df_all.to_excel(f'test_result_all/prediction_result_new/excel/{word_len}_{prediction_len}_{dropout}_new.xlsx', index=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

file_path = '../data/'
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

for prediction_len in [1]:
    print(f'prediction length: {prediction_len}')
    # pretrain_path = f'../checkpoints/pretrain/word-{word_len}-{dropout}/'
    # finetune_path = f'../checkpoints/finetune/prediction/word-{word_len}-p-{prediction_len}-dropout{dropout}-new/'
    finetune_path = f'../checkpoints/finetune/prediction/lstm-p{prediction_len}-dropout{dropout}-step36/'

    train_file = file_path + 'train%dsmoothed.csv' % (seq_len+prediction_len)
    valid_file = file_path + 'valid%dsmoothed.csv' % (seq_len+prediction_len)
    test_file = file_path + 'test%dsmoothed.csv' % (seq_len+prediction_len)
    # train_file = file_path + f'train_step36_pre{prediction_len}_smoothed.csv'
    # valid_file = file_path + f'valid_step36_pre{prediction_len}_smoothed.csv'
    # test_file = file_path + f'test_step36_pre{prediction_len}_smoothed.csv'

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

    # tvtsbert = TVTSBERT(word_len=word_len,
    #                     hidden=hidden_size,
    #                     n_layers=layers,
    #                     pe_window=pe_window,
    #                     attn_heads=attn_heads,
    #                     dropout=dropout)
    lstm = LSTM(input_size=num_features,
                hidden_size=hidden_size,
                num_layers=layers,
                prediction_len=prediction_len,
                )

    # tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
    # tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

    # tvtsbert.load_state_dict(torch.load(tvtsbert_path))


    # finetuner = TVTSBERTFTPredictor(tvtsbert, num_features=num_features,
    #                                 seq_len=seq_len, prediction_len=prediction_len,
    #                                 word_len=word_len, train_dataloader=train_dataloader,
    #                                 valid_dataloader=valid_dataloader)
    finetuner = LSTMPredictor(lstm, seq_len=seq_len, prediction_len=prediction_len,
                              hidden=hidden_size, num_features=num_features, layers=layers, lr=learning_rate,
                              train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

    # Test: 重新加载finetune的模型
    print("\n" * 5)
    print("Testing TVTS-BERT Finetune Predictor...")


    finetuner.load(finetune_path)

    # test需要输入参数i:第几个epoch(< 31)
    # 参数j:第几个样本(< 32)

    # prediction_result_list, prediction_target_list = finetuner.test(test_dataloader)
    prediction_result_list, input289_list = finetuner.test(test_dataloader)


    # test2_plot(prediction_result_list, input289_list, prediction_len=prediction_len)
    save_results(prediction_result_list, input289_list,
        word_len=word_len, prediction_len=prediction_len, dropout=dropout)
    print('----------\n')

