import numpy as np
from tqdm import tqdm
# import utils
import torch
import torch.nn as nn
# from torch.optim import Adam
from torch.utils.data import DataLoader
from finetune.finetune_dataset import FinetuneDataset
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn import SVC

train_file = file_path + 'ff_train_72.csv'
valid_file = file_path + 'ff_valid_72.csv'
test_file = file_path + 'ff_test_72.csv'
# train_file = file_path + 'ff_train_96.csv'
# valid_file = file_path + 'ff_valid_96.csv'
# test_file = file_path + 'ff_test_96.csv'
