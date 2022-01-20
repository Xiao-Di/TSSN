# -*-coding:utf-8-*-
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.tvts_bert import TVTSBERT
from finetune.ft_complementation_model import TVTSBERTFTComplementation

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class TVTSBERTFTComplementator:
    def __init__(self, tvtsbert: TVTSBERT, num_features, seq_len, word_len, mask_len,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float = 1e-4, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.tvtsbert = tvtsbert
        self.model = TVTSBERTFTComplementation(tvtsbert, seq_len, word_len, mask_len).to(self.device)
        # self.num_classes = num_classes
        self.seq_len = seq_len
        self.word_len = word_len
        self.mask_len = mask_len

        # 多gpu并行操作
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model pretraining" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

        # 每次finetune之前改一下writer的地址
        writer_path = '../runs/finetune/ft_complementation_%d_%d_0' % (self.word_len, self.mask_len)
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        self.writer = SummaryWriter(writer_path)
        self.log_freq = log_freq

    def train(self, epoch):

        # 进度条
        data_iter = tqdm(enumerate(self.train_dataloader),
                         desc="EP_%s:%d" % ("train", epoch),
                         total=len(self.train_dataloader),
                         bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            # print('shape of bert_input:', data['bert_input'].shape)
            # print('shape of bert_mask:', data['bert_mask'].shape)
            # print('shape of bert_target:', data['bert_target'].shape)
            # print('shape of loss_mask:', data['loss_mask'].shape)

            # 计算后一段时间位置处的预测值，并与真实值算loss:MSE
            finetune_complementation = self.model(data['bert_input'].float(),
                                                  data['bert_mask'].long())  # (288,1)
            # print('result: ', finetune_complementation.shape) # [batchsize, 288, 1]
            # print('target: ', data['bert_target'].shape) # [batchsize, 288, 1]


            loss = self.criterion(finetune_complementation,
                                  data['bert_target'].float())  # nn.MSELoss(reduction='none'), (288,1)
            mask = data['loss_mask'].unsqueeze(-1)  # (288,) -> (288,1)
            loss = (loss * mask.float()).sum() / mask.sum()  # 对48位需要预测对位置(全部序列)求loss的平均

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clipping) # 防止梯度爆炸
            self.optim.step()

            train_loss += loss.item()  # 取scalar，叠加

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i + 1),  # epoch的平均loss
                "loss": loss.item()  # iter的loss
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

        valid_loss = self._validate()
        self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

        # warmup
        # if epoch >= self.warmup_epochs:
        # self.optim_schedule.step()
        # self.writer.add_scalar('cosine_lr_decay', self.optim_schedule.get_lr()[0], global_step=epoch) # lr第一维

        print("EP%d, train_loss=%.5f, validation_loss=%.5f" % (epoch, train_loss, valid_loss))

        return train_loss, valid_loss

    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for data in self.valid_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                finetune_complementation = self.model(data['bert_input'].float(),
                                                      data['bert_mask'].long())

                # print(len(finetune_prediction[0])) # 84

                loss = self.criterion(finetune_complementation, data['bert_target'].float())
                mask = data['loss_mask'].unsqueeze(-1)
                loss = (loss * mask.float()).sum() / mask.sum()

                valid_loss += loss.item()
                counter += 1

            valid_loss /= counter

        self.model.train()
        return valid_loss

    def test(self, data_loader):
        max = 638.3
        min = 0.0

        with torch.no_grad():
            self.model.eval()

            complementation_result_list = [] # 288
            input288_list = [] # 288
            input48_list = []
            position_list = []

            print('num of data in test dataloader: ', len(data_loader))

            test_error = 0.0
            overall_error = 0.0

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                complementation_result = self.model(data['bert_input'].float(),
                                                    data['bert_mask'].long())  # [bs, 12, 1]
                input288 = data['bert_target'].float()  # [bs, 288, 1]
                input48 = data['bert_target48'].float() # [bs, 12, 1]
                position = data['position'].float() # [bs, 288]

                # 反归一化
                complementation_result_inverse = ((max - min) * complementation_result + min).squeeze()  # [256,288]
                input288_inverse = ((max - min) * input288 + min).squeeze()  # [256,288]
                input48_inverse = ((max - min) * input48 + min).squeeze()  # [256,288]

                complementation_result_list.append(complementation_result_inverse.cpu().numpy()) # 288点完整的
                input288_list.append(input288_inverse.cpu().numpy())                     # 完整的一串288点输入
                input48_list.append(input48_inverse.cpu().numpy())                       # 288点，只有中间48点是真实值，其余位置是0
                position_list.append(position.cpu().numpy())

            print(np.array(complementation_result_list[:-1]).shape) # [88,256,88]
            print(np.array(input288_list[:-1]).shape)               # [88,256,88]
            print(np.array(input48_list[:-1]).shape)                # [88,256,88]
            print(np.array(position_list[:-1]).shape)               # [88,256,88]

            result12 = {
                'Comple12': np.array(complementation_result_list[:-1]).flatten(),
                'Input12': np.array(input48_list[:-1]).flatten(),
            }
            result288 = {
                'Input288': np.array(input288_list[:-1]).flatten()[:1048320],
                'Position': np.array(position_list[:-1]).flatten()[:1048320],
            }
            df12 = pd.DataFrame(result12)
            df288 = pd.DataFrame(result288)
            df_save = pd.concat([df288, df12], axis=1)

            # result_df = pd.DataFrame({'Complementation': np.array(complementation_result_list[:-1]).flatten(),
            #                           'Input288': np.array(input288_list[:-1]).flatten(),
            #                           'Input48': np.array(input48_list[:-1]).flatten(),
            #                           'Position': np.array(position_list[:-1]).flatten()})
            print('Saving to test_result_all/complement_plot_1227/Complement_K%d_C%d111.csv' % (self.word_len, self.mask_len))
            df_save.to_csv(f'/Users/gengyunxin/Documents/项目/traffic_model/TVTS_9.28/finetune/test_result_all/complement_plot_1227/complement_K{self.word_len}_C{self.mask_len}_i216.csv')

        self.model.train()
        return test_error

    def save(self, epoch, file_path):
        output_path = file_path + "complement_checkpoint.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
        }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + "complement_checkpoint.tar"

        # checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(input_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model Loaded from:" % epoch, input_path)
        return input_path


