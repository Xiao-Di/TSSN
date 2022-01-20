import torch
from torch.utils.data import DataLoader
import sys
import os
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

word_len = 2
file_path = '../data/'
pretrain_path = '../checkpoints/pretrain/word-%d-0.1/' % word_len # the storage path of the pretrained model
finetune_path = '../checkpoints/finetune/classification/word-%d/' % word_len # the output directory where the finetuning checkpoints written
if not os.path.exists(finetune_path):
    os.makedirs(finetune_path)

max_length = 288
num_features = 1
num_classes = 2 # 分类任务数
pe_window = 288

epochs = 100
batch_size = 64
hidden_size = 64
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

# train_file = file_path + 'ff_train_72.csv'
# valid_file = file_path + 'ff_valid_72.csv'
# test_file = file_path + 'ff_test_72.csv'
train_file = file_path + 'train_classify.csv'
valid_file = file_path + 'valid_classify.csv'
test_file = file_path + 'test_classify.csv'

print("Loading data sets...")
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
                              seq_len=max_length, word_len=word_len,
                              train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader)

print("Finetuning TVTS-BERT for Classification...")
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

# fig = plt.figure()
# ax1 = plt.subplot(221)
# ax1.plot(list(range(epochs)), train_loss_list)
# ax1.set_title('train loss/finetune')
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('train_loss')

# ax2 = plt.subplot(222)
# ax2.plot(list(range(epochs)), train_overall_acc_list)
# ax2.set_title('train acc/finetune')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('train_acc')

# ax3 = plt.subplot(223)
# ax3.plot(list(range(epochs)), valid_loss_list)
# ax3.set_title('valid loss/finetune')
# ax3.set_xlabel('epoch')
# ax3.set_ylabel('valid_loss')

# ax4 = plt.subplot(224)
# ax4.plot(list(range(epochs)), valid_overall_acc_list)
# ax4.set_title('valid acc/finetune')
# ax4.set_xlabel('epoch')
# ax4.set_ylabel('valid_acc')

# plt.savefig('4.30_1_finetune72_result.png')




# Test: 重新加载finetune的模型
print("\n" * 5)
print("Testing TVTS-BERT...")
finetuner.load(finetune_path)
test_overall_acc, test_avg_acc, test_kappa, classification_result, classification_target = finetuner.test(test_dataloader)
precision = precision_score(classification_target, classification_result, average='weighted')
recall = recall_score(classification_target, classification_result, average='weighted')
F1 = f1_score(classification_target, classification_result, average='weighted')

print("test_overall_acc=%.2f, test_avg_acc=%.4f, test_kappa=%.3f" % (test_overall_acc, test_avg_acc, test_kappa))
print('\n')
print('Precision=%.3f, Recall=%.3f, F1_score=%.3f' % (precision, recall, F1))