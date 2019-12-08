import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time

from lenet.model.lenet import LeNet


class MLELoss(nn.Module):

    def __init__(self):
        super(MLELoss, self).__init__()

    def forward(self, predict, label):
        one_hot = torch.zeros_like(predict).long()
        one_hot.scatter_(1, label.unsqueeze(1), 1)
        return predict.mul(one_hot).sum() / predict.size(0)


def get_learning_rate(epoch=1):
    if epoch < 3:
        return 0.0005
    elif epoch < 6:
        return 0.0002
    elif epoch < 9:
        return 0.0001
    elif epoch < 13:
        return 0.00005
    else:
        return 0.00001


def load_data():
    # 数据预处理：1. 将输入图片的大小宽展为 32x32；2. 进行标准化，转化为均值为 0，方差为 1 的数据
    # transforms.Normalize((0.1307,), (0.3081,)) 的讨论参见：
    #       https://discuss.pytorch.org/t/normalization-in-the-mnist-example
    data_transform = transforms.Compose([transforms.Pad(2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=data_transform)
    train_data_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_set = datasets.MNIST('./data', train=False, transform=data_transform)
    test_data_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_data_loader, test_data_loader


def set_leaning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_num(predict, label):
    predict = np.argmin(predict.detach().numpy(), 1)
    label = label.numpy()
    return (predict == label).sum()


def train():
    train_data_loader, test_data_loader = load_data()

    net = LeNet()
    loss = MLELoss()
    epochs = 20
    optimizer = optim.SGD(net.parameters(), lr=0.0005)
    start_time = time.time()
    print("start time {}".format(start_time))
    for epoch in range(epochs):
        net.train()
        set_leaning_rate(optimizer, get_learning_rate(epoch + 1))
        for idx, (images, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            predicts = net.forward(images)
            l = loss(predicts, labels)
            l.backward()
            optimizer.step()
        net.eval()
        total_accuracy_num = 0
        with torch.no_grad():
            for idx, (images, labels) in enumerate(train_data_loader):
                predicts = net.forward(images)
                total_accuracy_num += accuracy_num(predicts, labels)
        print("epoch: {0}, accuracy rate: {1:.2%}".format(epoch, total_accuracy_num / len(train_data_loader) / 256))
        print("epoch time {}".format(time.time()))

    test_accuracy_num = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_data_loader):
            predicts = net.forward(images)
            test_accuracy_num += accuracy_num(predicts, labels)
    print("test accuracy rate: {:.2%}".format(test_accuracy_num / len(test_data_loader) / 256))


if __name__ == "__main__":
    train()
