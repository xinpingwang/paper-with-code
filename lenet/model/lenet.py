import torch
from torch import nn

from lenet.utils.rbf_weight import RBF_WEIGHT
from lenet.model.layers import Sampling2d, DropoutConv2d, RBF


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
        self.conv3 = DropoutConv2d(6, 16, mapping, 5)
        self.s4 = Sampling2d(16, 2, 2)
        self.conv5 = nn.Conv2d(16, 120, 5)
        # an affine operation: y = Wx + b
        self.fc6 = nn.Linear(120, 84)
        self.rbf = RBF(84, 10, init_weight=RBF_WEIGHT)

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
