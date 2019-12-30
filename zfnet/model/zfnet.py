import torch
from torch import nn


class ZFNet(nn.Module):

    def __init__(self):
        super(ZFNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, k=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(96, 256, 5, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, k=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = ZFNet()
    total_param_num = torch.tensor(0, dtype=torch.long)
    for param in net.parameters():
        a = torch.tensor(param.shape, dtype=torch.long)
        total_param_num += a.prod()
    print('There are {} parameters in the model.'.format(total_param_num.item()))
