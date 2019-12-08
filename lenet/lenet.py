import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time


class Sampling2d(nn.Module):

    def __init__(self, in_channel, kernel_size, stride, padding=0):
        super(Sampling2d, self).__init__()
        # 降采样是对采样窗口内的数据求和，然后乘以一个可训练的系数，加上一个可训练的偏差
        # 其中求和过程可以使用平均池化间接来实现
        self.avg_pool2d = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.in_channel = in_channel
        # 初始化通道数个权重和偏差
        self.weights = nn.Parameter(torch.randn(in_channel))
        self.biases = nn.Parameter(torch.randn(in_channel))

    def forward(self, x):
        # x 的 shape 为 (n, c, h, w)
        x = self.avg_pool2d(x)
        outs = []
        for i in range(self.in_channel):
            out = x[:, i] * self.weights[i] + self.biases[i]
            outs.append(out.unsqueeze(1))
        return torch.cat(outs, 1)


class DropoutConv2d(nn.Module):

    # mapping 指定输入和输出的映射关系，是一个二维数组
    def __init__(self, in_channels, out_channels, mapping, kernel_size, stride=1, padding=0, bias=True):
        super(DropoutConv2d, self).__init__()
        # 保证 mapping 的长度和 out_channels 一致
        assert out_channels == len(mapping)
        self.mapping = mapping
        self.convs = {}
        for i in range(len(mapping)):
            conv = nn.Conv2d(len(mapping[i]), 1, kernel_size, stride=stride, padding=padding, bias=bias)
            module_name = 'conv{}'.format(i)
            self.convs[module_name] = conv
            # 通过 add_module 将 conv 中的参数注册到当前模块中
            self.add_module(module_name, conv)

    def forward(self, x):
        out = []
        for i in range(len(self.mapping)):
            in_channels = torch.tensor(self.mapping[i], dtype=torch.long)
            in_tensors = x.index_select(1, in_channels)
            conv_out = self.convs['conv{}'.format(i)](in_tensors)
            out.append(conv_out)

        return torch.cat(out, 1)


class MLELoss(nn.Module):

    def __init__(self):
        super(MLELoss, self).__init__()

    def forward(self, predict, label):
        one_hot = torch.zeros_like(predict).long()
        one_hot.scatter_(1, label.unsqueeze(1), 1)
        return predict.mul(one_hot).sum() / predict.size(0)


class RBF(nn.Module):

    def __init__(self, in_features, out_features, init_weight=None):
        super(RBF, self).__init__()
        if init_weight is not None:
            self.weight = torch.tensor(init_weight)
        else:
            self.weight = torch.rand(in_features, out_features)

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = (x - self.weight).pow(2).sum(-1)
        return x


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.s2 = Sampling2d(6, 2, 2)
        mapping = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
                   [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],
                   [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
                   [0, 1, 2, 3, 4, 5]]
        rbf_weight = [
            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, +1, +1, +1, -1, -1] +
            [-1, +1, +1, -1, +1, +1, -1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [-1, +1, +1, -1, +1, +1, -1] +
            [-1, -1, +1, +1, +1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, +1, +1, +1, -1, -1] +
            [-1, +1, +1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, +1, +1, -1] +
            [-1, -1, +1, +1, +1, -1, -1] +
            [-1, +1, +1, -1, -1, -1, -1] +
            [+1, +1, -1, -1, -1, -1, -1] +
            [+1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [+1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, +1, +1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, +1, +1, -1, -1, +1, +1] +
            [-1, +1, +1, -1, -1, +1, +1] +
            [+1, +1, +1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, +1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1],

            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [+1, +1, +1, +1, +1, +1, +1] +
            [+1, +1, -1, -1, -1, -1, -1] +
            [+1, +1, -1, -1, -1, -1, -1] +
            [-1, +1, +1, +1, +1, -1, -1] +
            [-1, -1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, -1, +1, +1, +1, +1, -1] +
            [-1, +1, +1, -1, -1, -1, -1] +
            [+1, +1, -1, -1, -1, -1, -1] +
            [+1, +1, -1, -1, -1, -1, -1] +
            [+1, +1, +1, +1, +1, +1, -1] +
            [+1, +1, +1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, +1, -1, -1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [+1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, +1, +1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, -1, +1, +1, -1, -1] +
            [-1, -1, +1, +1, -1, -1, -1] +
            [-1, -1, +1, +1, -1, -1, -1] +
            [-1, -1, +1, +1, -1, -1, -1] +
            [-1, -1, +1, +1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, +1, +1, +1, +1, +1, -1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1],

            [-1, +1, +1, +1, +1, +1, -1] +
            [+1, +1, -1, -1, +1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, -1, +1, +1] +
            [+1, +1, -1, -1, +1, +1, +1] +
            [-1, +1, +1, +1, +1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, -1, +1, +1] +
            [-1, -1, -1, -1, +1, +1, -1] +
            [-1, +1, +1, +1, +1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1] +
            [-1, -1, -1, -1, -1, -1, -1]
        ]
        self.conv3 = DropoutConv2d(6, 16, mapping, 5)
        self.s4 = Sampling2d(16, 2, 2)
        self.conv5 = nn.Conv2d(16, 120, 5)
        # an affine operation: y = Wx + b
        self.fc6 = nn.Linear(120, 84)
        self.rbf = RBF(84, 10, init_weight=rbf_weight)

    def forward(self, x):
        def active(a):
            return 2 * 1.7259 * torch.sigmoid(4 / 3 * a) - 1

        x = active(self.conv1(x))
        x = active(self.s2(x))
        x = active(self.conv3(x))
        x = active(self.s4(x))
        x = active(self.conv5(x))
        x = torch.squeeze(x)
        x = active(self.fc6(x))
        x = self.rbf(x)
        return x


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
