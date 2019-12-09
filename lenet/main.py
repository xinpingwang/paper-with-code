import os
import torch
from torch import nn
from torch import optim

import numpy as np

from lenet.model.lenet import LeNet
from lenet.utils import data_loader, trainer


class MLELoss(nn.Module):

    def __init__(self):
        super(MLELoss, self).__init__()

    def forward(self, predict, label):
        one_hot = torch.zeros_like(predict).long()
        one_hot.scatter_(1, label.unsqueeze(1), 1)
        return predict.mul(one_hot).sum() / predict.size(0)


def get_learning_rate(epoch):
    lr = 0
    if epoch < 3:
        lr = 0.0005
    elif epoch < 6:
        lr = 0.0002
    elif epoch < 9:
        lr = 0.0001
    elif epoch < 13:
        lr = 0.00005
    else:
        lr = 0.00001
    return lr


def accuracy_num(predict, label):
    predict = np.argmin(predict.detach().numpy(), 1)
    label = label.numpy()
    return int((predict == label).sum())


MODEL_FILE = 'LeNet.pth'
BATCH_SIZE = 256
EPOCHS = 20

train_data_loader, test_data_loader = data_loader.load_data('./data', batch_size=BATCH_SIZE)
model = LeNet()
criterion = MLELoss()

# LambdaLR 设置的学习速率是：初始学习速率 * lambda 函数返回的结果
optimizer = optim.SGD(model.parameters(), lr=1)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: get_learning_rate(epoch + 1))

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE))
    print('loaded model from {}'.format(MODEL_FILE))


trainer.train(model, train_data_loader, test_data_loader, criterion, optimizer, accuracy_num, EPOCHS,
              lr_scheduler=lr_scheduler, checkpoint=MODEL_FILE)
