from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(data_dir, batch_size=256):

    # 数据预处理：1. 将输入图片的大小宽展为 32x32；2. 进行标准化，转化为均值为 0，方差为 1 的数据
    # transforms.Normalize((0.1307,), (0.3081,)) 的讨论参见：
    #       https://discuss.pytorch.org/t/normalization-in-the-mnist-example
    data_transform = transforms.Compose([transforms.Pad(2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=data_transform)
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = datasets.MNIST(data_dir, train=False, transform=data_transform)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader
