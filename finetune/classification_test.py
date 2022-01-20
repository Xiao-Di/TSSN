import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from model.tvts_bert import TVTSBERT
from finetune import TVTSBERTFineTuner
from finetune_dataset import FinetuneDataset
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def avg_acc(matrix):
    correct = np.diag(matrix)
    all = matrix.sum(axis=0)
    acc = correct / all
    avg_acc = np.average(acc)
    return avg_acc


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

file_path = '../data/'

pretrain_path = '../checkpoints/pretrain/word6/' # the storage path of the pretrained model
finetune_path = '../checkpoints/finetune/' # the output directory where the finetuning checkpoints written

# max_length = 72
max_length = 96
num_features = 1

# num_classes = 2 # 分类任务数
num_classes = 3 # 分类任务数
word_len = 6

pe_window = 288

epochs = 100
batch_size = 128
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

# train_file = file_path + 'ff_train_72.csv'
# valid_file = file_path + 'ff_valid_72.csv'
# test_file = file_path + 'ff_test_72.csv'
train_file = file_path + 'ff_train_96.csv'
valid_file = file_path + 'ff_valid_96.csv'
test_file = file_path + 'ff_test_96.csv'

print("Lodaing data sets...")
train_dataset = FinetuneDataset(file_path=train_file,
                                num_features=num_features,
                                seq_len=max_length,
                                word_len=word_len)
valid_dataset = FinetuneDataset(file_path=valid_file,
                                num_features=num_features,
                                seq_len=max_length,
                                word_len=word_len)
test_dataset = FinetuneDataset(file_path=test_file,
                               num_features=num_features,
                               seq_len=max_length,
                               word_len=word_len)

print("Creating dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)

print("Initializing TVTS-BERT...")
tvtsbert = TVTSBERT(word_len=word_len,
                    hidden=hidden_size,
                    n_layers=layers,
                    pe_window=pe_window,
                    attn_heads=attn_heads,
                    dropout=dropout)

print("Loading pretrained model parameters...")
tvtsbert_path = pretrain_path + "checkpoint.bert.pth" # 地址加一下epoch
# tvtsbert.load_state_dict(torch.load(tvtsbert_path))
tvtsbert.load_state_dict(torch.load(tvtsbert_path, map_location=torch.device('cpu')))

print("Creating downstream classification task FineTuner...")
finetuner = TVTSBERTFineTuner(tvtsbert, num_classes=num_classes,
                              train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader)



print("\n" * 5)
print("Testing TVTS-BERT...")
finetuner.load(finetune_path)
test_overall_acc, test_avg_acc, test_kappa, classification_result, classification_target = finetuner.test(test_dataloader)
precision = precision_score(classification_target, classification_result, average='weighted')
recall = recall_score(classification_target, classification_result, average='weighted')
F1 = f1_score(classification_target, classification_result, average='weighted')

print("test_overall_acc=%.2f, test_avg_acc=%.3f, test_kappa=%.3f" % (test_overall_acc, test_avg_acc, test_kappa))
print('\n')
print('Precision=%.3f, Recall=%.3f, F1_score=%.3f' % (precision, recall, F1))