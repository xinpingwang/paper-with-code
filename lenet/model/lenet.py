import torch
from torch import nn

from lenet.utils.rbf_weight import RBF_WEIGHT
from lenet.model.layers import Sampling2d, DropoutConv2d, RBF


class LeNet(nn.Module):

    def __init__(self, pooling=Sampling2d, active=None, softmax=False):
        """
        args:
            pooling: 指定降采样方式
            active: 激活函数，默认使用 LeNet 论文中的 tanh
            softmax: 如果为 True 则最后一层输出为全联接加 softmax 结果，否则为 RBF 输出
        """
        super(LeNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.s2 = pooling(6, 2, 2)
        mapping = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        self.conv3 = DropoutConv2d(6, 16, mapping, 5)
        self.s4 = pooling(16, 2, 2)
        self.conv5 = nn.Conv2d(16, 120, 5)
        # an affine operation: y = Wx + b
        self.fc6 = nn.Linear(120, 84)
        if softmax:
            self.fc7 = nn.Linear(84, 10)
        else:
            self.rbf = RBF(84, 10, init_weight=RBF_WEIGHT)
        self.softmax = softmax

        if active is not None:
            self.active = active
        else:
            self.active = lambda x: 1.7259 * (2 * torch.sigmoid(4 / 3 * x) - 1)

    def forward(self, x):
        x = self.active(self.conv1(x))
        x = self.active(self.s2(x))
        x = self.active(self.conv3(x))
        x = self.active(self.s4(x))
        x = self.active(self.conv5(x))
        x = torch.squeeze(x)
        x = self.active(self.fc6(x))
        if self.softmax:
            x = self.fc7(x)
            x = x.softmax(1)
        else:
            x = self.rbf(x)
        return x
