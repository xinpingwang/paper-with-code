import torch
from torch import nn


class Sampling2d(nn.Module):

    def __init__(self, in_channel, kernel_size, stride, padding=0):
        super(Sampling2d, self).__init__()
        # 降采样是对采样窗口内的数据求和，然后乘以一个可训练的系数，加上一个可训练的偏差
        # 其中求和过程可以使用平均池化间接来实现
        self.avg_pool2d = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.in_channel = in_channel
        # 初始化通道数个权重和偏差
        self.weights = nn.Parameter(torch.randn(in_channel), requires_grad=True)
        self.biases = nn.Parameter(torch.randn(in_channel), requires_grad=True)

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
        # 保证 mapping 的 shape 为 (in_channels, out_channels)
        assert in_channels == len(mapping) and out_channels == len(mapping[0])
        mapping = torch.tensor(mapping, dtype=torch.long)
        self.register_buffer('mapping', mapping)
        self.convs = {}
        for i in range(self.mapping.size(1)):
            conv = nn.Conv2d(self.mapping[:, i].sum().item(), 1, kernel_size, stride=stride, padding=padding, bias=bias)
            module_name = 'conv{}'.format(i)
            self.convs[module_name] = conv
            # 通过 add_module 将 conv 中的参数注册到当前模块中
            self.add_module(module_name, conv)

    def forward(self, x):
        out = []
        for i in range(self.mapping.size(1)):
            in_channels = self.mapping[:, i].nonzero().squeeze()
            in_tensors = x.index_select(1, in_channels)
            conv_out = self.convs['conv{}'.format(i)](in_tensors)
            out.append(conv_out)

        return torch.cat(out, 1)


class RBF(nn.Module):

    def __init__(self, in_features, out_features, init_weight=None):
        super(RBF, self).__init__()
        if init_weight is not None:
            self.register_buffer('weight', torch.tensor(init_weight))
        else:
            # 如果没有提供初始权重，则随机初始化的权重应该是可训练的
            self.weight = nn.Parameter(torch.rand(in_features, out_features), requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2)
        return x
