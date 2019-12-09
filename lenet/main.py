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
        return predict.gather(1, label.view(-1, 1)).sum() / predict.size(0)


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
    predict = np.argmin(predict.detach().cpu().numpy(), 1)
    label = label.cpu().numpy()
    return int((predict == label).sum())


MODEL_FILE = 'LeNet.pth'
BATCH_SIZE = 256
EPOCHS = 20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data_loader, test_data_loader = data_loader.load_data('./data', batch_size=BATCH_SIZE)
model = LeNet()
model.to(device)
criterion = MLELoss()

# LambdaLR 设置的学习速率是：初始学习速率 * lambda 函数返回的结果
optimizer = optim.SGD(model.parameters(), lr=1)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: get_learning_rate(epoch + 1))

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE), map_location=device)
    print('loaded model from {}'.format(MODEL_FILE))


train_acc_rates, val_acc_rates = trainer.train(model, train_data_loader, test_data_loader, criterion, optimizer,
                                               accuracy_num, EPOCHS, lr_scheduler=lr_scheduler, checkpoint=MODEL_FILE,
                                               device=device)
