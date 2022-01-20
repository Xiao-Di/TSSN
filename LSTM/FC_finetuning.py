import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
# from model.tvts_bert import TVTSBERT
# from finetune import TVTSBERTFineTuner
from FC import FC
from FC_finetune import FCFineTuner
from finetune.finetune_dataset import FinetuneDataset
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

file_path = '../data/'

# pretrain_path = '../checkpoints/pretrain/word1/' # the storage path of the pretrained model
finetune_path = '../checkpoints/finetune/' # the output directory where the finetuning checkpoints written

max_length = 288
num_features = 1

num_classes = 2 # 分类任务数
# num_classes = 3 # 分类任务数
word_len = 1

epochs = 100
batch_size = 128
hidden_size = 64
n_hidden_1 = 30
n_hidden_2 = 10
layers = 3
# attn_heads = 8
learning_rate = 1e-3
dropout = 0.1

# train_file = file_path + 'ff_train_72.csv'
# valid_file = file_path + 'ff_valid_72.csv'
# test_file = file_path + 'ff_test_72.csv'
train_file = file_path + 'train_classify.csv'
valid_file = file_path + 'valid_classify.csv'
test_file = file_path + 'test_classify.csv'

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

print("Initializing FC...")
fc = FC(in_dim=max_length*num_features,
        n_hidden_1=n_hidden_1,
        n_hidden_2=n_hidden_2,
        out_dim=num_classes)

# print("Loading pretrained model parameters...")
# tvtsbert_path = pretrain_path + "checkpoint.bert.pth" # 地址加一下epoch
# tvtsbert.load_state_dict(torch.load(tvtsbert_path))

print("Creating downstream classification task FineTuner...")
finetuner = FCFineTuner(fc, num_classes=num_classes,
                          num_features=num_features,
                          in_dim=max_length*num_features,
                          out_dim=num_classes,
                          n_hidden_1=n_hidden_1,
                          n_hidden_2=n_hidden_2,
                          layers=layers,
                          lr=learning_rate,
                          train_dataloader=train_dataloader,
                          valid_dataloader=valid_dataloader)

print("Finetuning LSTM for Classification...")
overall_acc = 0
train_loss_list = []
train_overall_acc_list = []
valid_loss_list = []
valid_overall_acc_list = []

for epoch in range(epochs):
    train_loss, train_overall_acc, train_kappa, valid_loss, valid_overall_acc, valid_kappa = finetuner.train(epoch)
    if overall_acc < valid_overall_acc:
        overall_acc = valid_overall_acc
        finetuner.save(epoch, finetune_path)

    train_loss_list.append(train_loss)
    train_overall_acc_list.append(train_overall_acc)
    valid_loss_list.append(valid_loss)
    valid_overall_acc_list.append(valid_overall_acc)



# Test: 重新加载finetune的模型
print("\n" * 5)
print("Testing FC Classification...")
finetuner.load(finetune_path)
test_overall_acc, test_avg_acc, test_kappa, classification_result, classification_target = finetuner.test(test_dataloader)

precision = precision_score(classification_target, classification_result, average='weighted')
recall = recall_score(classification_target, classification_result, average='weighted')
F1 = f1_score(classification_target, classification_result, average='weighted')
cohen_kappa = cohen_kappa_score(classification_target, classification_result)

print("test_overall_acc=%.2f, test_avg_acc=%.4f, test_kappa=%.3f" % (test_overall_acc, test_avg_acc, cohen_kappa))
print('Precision=%.3f, Recall=%.3f, F1_score=%.3f' % (precision, recall, F1))