import torch
from torch.utils.data import DataLoader
# from torchsummary import summary
import sys
sys.path.append("..")

from LSTM_prediction_model import LSTM
from LSTM_predict import LSTMPredictor
from finetune.finetune_predict_dataset import FinetunePredictDataset
import numpy as np
import random
from matplotlib import pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def fig_plot(prediction_result_list, prediction_target_list, batch, sample,
#              seq_len=288, prediction_len=12):
#     plt.figure()
#     y1 = prediction_result_list[batch][sample]
#     y2 = prediction_target_list[batch][sample]
#     # x1 = len(prediction_target[72:84])
#     x2 = len(y2)
#     plt.plot(list(range(x2)), np.append(y2[:seq_len].cpu(), y1[seq_len:seq_len+prediction_len].cpu()),
#             c='r', label='prediction result')
#     plt.plot(list(range(x2)), y2.cpu(), c='b', label='target')
#     plt.plot(list(range(x2)), y1.cpu(), c='g', label='prediction result all')
#     plt.title('batch%d-sample%d' % (batch, sample))
#     plt.legend(['prediction_result', 'prediction_target'])
#     plt.savefig('4.16finetune_predict_0_batch%d_sample%d.png' % (batch, sample))

setup_seed(123)

file_path = '../data/'

finetune_path = '../checkpoints/finetune/' # the output directory where the finetuning checkpoints written

seq_len = 288
prediction_len = 1
num_features = 1
word_len = 1 # fixed

epochs = 30
batch_size = 256
hidden_size = 64
layers = 3
learning_rate = 2e-5
dropout = 0.1

train_file = file_path + 'train289smoothed.csv'
valid_file = file_path + 'valid289smoothed.csv'
test_file = file_path + 'test289smoothed.csv'


print("Lodaing data sets...")
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

print("Creating dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, pin_memory=True,
                                batch_size=batch_size, drop_last=False)

print("Initializing LSTM...")
lstm = LSTM(input_size=num_features,
            hidden_size=hidden_size,
            num_layers=layers,
            prediction_len=prediction_len,
            )

# summary(tvtsbert, (64,1,144,144), (64,1,143,144))

# print("Loading pretrained model parameters...")
# tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
# tvtsbert.load_state_dict(torch.load(tvtsbert_path))
# tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

print("Creating downstream prediction task FineTuner...")
finetuner = LSTMPredictor(lstm, seq_len=seq_len, prediction_len=prediction_len,
                          hidden=hidden_size, num_features=num_features, layers=layers, lr=learning_rate,
                          train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)


print("Finetuning LSTM for Prediction...")
for epoch in range(epochs):
    train_loss, valid_loss = finetuner.train(epoch)
    finetuner.save(epoch, finetune_path)


