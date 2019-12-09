import numpy as np
import torch
from tqdm import tqdm


def train(model, train_data_loader, test_data_loader, criterion, optimizer, accuracy, epochs,
          lr_scheduler=None, checkpoint=None, verbose=True):
    train_acc_rates, val_acc_rates = [], []
    for epoch in range(epochs):
        if verbose:
            print('epoch {0}:'.format(epoch + 1))
        # train phase
        model.train()
        total_train_num = 0
        train_accuracy_num = 0
        if verbose:
            print('\ttraining ...')
        for images, labels in tqdm(train_data_loader):
            optimizer.zero_grad()
            predicts = model(images)
            # 计算损失函数，并进行反向传播
            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()
            # 统计总数和预测正确的数量
            total_train_num += len(images)
            train_accuracy_num += accuracy(predicts, labels)
        train_acc_rate = train_accuracy_num / total_train_num
        train_acc_rates.append(train_acc_rate)
        if verbose:
            print("\t train accuracy: {:.2%}".format(train_acc_rate))

        # 更新学习速率
        if lr_scheduler is not None:
            lr_scheduler.step()

        # val phase
        model.eval()
        total_val_num = 0
        val_accuracy_num = 0
        with torch.no_grad():
            if verbose:
                print('\t validating ...')
            for images, labels in tqdm(test_data_loader):
                predicts = model(images)
                # 统计总数和预测正确的数量
                total_val_num += len(images)
                val_accuracy_num += accuracy(predicts, labels)
        val_acc_rate = val_accuracy_num / total_val_num
        val_acc_rates.append(val_acc_rate)
        if verbose:
            print("\t val accuracy: {:.2%}".format(val_acc_rate))

        if checkpoint is not None:
            if len(val_acc_rates) == 0 or np.argmax(val_acc_rates) == len(val_acc_rates) - 1:
                torch.save(model.state_dict(), checkpoint)
    return train_acc_rates, val_acc_rates
