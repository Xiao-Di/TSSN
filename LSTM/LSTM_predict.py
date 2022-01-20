import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from LSTM_prediction_model import LSTM

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class LSTMPredictor:
    def __init__(self, lstm: LSTM, seq_len, prediction_len, hidden, num_features,
                 layers, train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float=1e-4, with_cuda: bool=True,
                 cuda_devices=None, log_freq: int=10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = LSTM(num_features, hidden, layers, prediction_len).to(self.device)
        # self.num_classes = num_classes
        self.seq_len = seq_len
        self.prediction_len = prediction_len

        # 多gpu并行操作
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUs for model pretraining" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

        # 每次finetune之前改一下writer的地址
        self.writer = SummaryWriter('../runs/2021.4.30-lstm_predict1new')
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
            finetune_prediction = self.model(data['bert_input'].float())

            loss = self.criterion(finetune_prediction, data['bert_target'].float()) # nn.MSELoss(reduction='none'), (84,1)
            mask = data['loss_mask'].unsqueeze(-1) # (12,) -> (12,1)
            loss = (loss * mask.float()).sum() / mask.sum() # 对后12位需要预测对位置(全部序列)求loss的平均

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clipping) # 防止梯度爆炸
            self.optim.step()

            train_loss += loss.item() # 取scalar，叠加

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i+1), # epoch的平均loss
                "loss": loss.item() # iter的loss
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

                finetune_prediction = self.model(data['bert_input'].float())

                # print(len(finetune_prediction[0])) # 84

                loss = self.criterion(finetune_prediction, data['bert_target'].float())
                mask = data['loss_mask'].unsqueeze(-1)
                loss = (loss * mask.float()).sum() / mask.sum()

                valid_loss += loss.item()
                counter += 1

            valid_loss /= counter

        self.model.train()
        return valid_loss


    def test(self, data_loader):
        """
            取test_dataloader(1000条样本)的第i个batch中的第j个样本画图对比prediction_result和target
        """
        max = 369.78
        min = 0.00

        with torch.no_grad():
            self.model.eval()

            prediction_result_list = []
            prediction_target_list = []
            input300_list = []

            print('num of data in test dataloader: ', len(data_loader))
            counter = 0
            test_error = 0.0
            overall_error = 0.0

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}
                prediction_result = self.model(data['bert_input'].float()) # [128, 1, 1]
                # print('prediction result; ', prediction_result.shape)
                prediction_target = data['bert_target'].float() # 1 [128, 1, 1]
                # print('target: ', prediction_target.shape)
                input300 = data['bert_input300'].float() # 289 [128, 289, 1]
                # print('input: ', input300.shape)

                # 反归一化
                prediction_result_inverse = ((max-min)*prediction_result + min).squeeze(-1) # [128, 1]
                # print('prediction inverse result; ', prediction_result_inverse.shape)
                prediction_target_inverse = ((max-min)*prediction_target + min).squeeze() # [128, 1]
                input300_inverse = ((max-min)*input300 + min).squeeze()  # [128, 1]
                # print('input_inverse: ', input300_inverse.shape)

                prediction_result_list.append(prediction_result_inverse)
                prediction_target_list.append(prediction_target_inverse)
                input300_list.append(input300_inverse)

            # print('length of prediction_target_list: ', len(prediction_target_list))
            # print('prediction_target_list[0] shape: ', prediction_target_list[0].shape)
            # print('prediction_target_list[-1] shape: ', prediction_target_list[-1].shape)
            # print('prediction_target_list: ', prediction_target_list)
            # print('input300_list: ', input300_list)
            # print('length of input300_list: ', len(input300_list))
            # print('input300[0] shape: ', input300_list[0].shape)
            # print('input300[-1] shape: ', input300_list[-1].shape)

        self.model.train()
        return prediction_result_list, prediction_target_list, input300_list


    def save(self, epoch, file_path):
        output_path = file_path + "lstm_predict_checkpoint.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
            }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + "lstm_predict_checkpoint.tar"

        checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
        # checkpoint = torch.load(input_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model Loaded from:" % epoch, input_path)
        return input_path


