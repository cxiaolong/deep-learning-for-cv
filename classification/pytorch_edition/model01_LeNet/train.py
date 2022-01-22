import ssl

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import LeNet

ssl.create_default_context()


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 50000张训练图片
    train_dataset = torchvision.datasets.CIFAR10(root='~/data/', download=False, train=True, transform=transform)
    # 10000张测试图片
    val_dataset = torchvision.datasets.CIFAR10(root='~/data/', download=False, train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=5000, shuffle=False, num_workers=0)
    val_data_iter = iter(val_dataloader)
    val_image, val_label = val_data_iter.next()

    net = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(10):
        running_loss = 0.0
        for step, data in enumerate(train_dataloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward+backward+optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # 每500个mini-batch打印一次
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]  # 取出第一个维度最大值的索引
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print(['%d, %5d train loss: %.3f, test accuracy: %.3f' % (
                        epoch + 1, step + 1, running_loss / 500, accuracy)])
                    running_loss = 0.0

    save_path = 'LeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    train()
