import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch, 3, 32, 32] --> [batch, 6, 28, 28]
        x = self.pool1(x)  # [batch, 6, 28, 28] --> [batch, 6, 14, 14]
        x = F.relu(self.conv2(x))  # [batch, 6, 14, 14] --> [batch, 16, 10, 10]
        x = self.pool2(x)  # [batch, 16, 10, 10] --> [batch, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)  # [batch, 16, 5, 5] --> [batch, 400]
        x = F.relu(self.fc1(x))  # [batch, 400] --> [batch, 120]
        x = F.relu(self.fc2(x))  # [batch, 120] --> [batch, 84]
        x = self.fc3(x)  # [batch, 84] --> [batch, 10]
        return x


# debug test
if __name__ == '__main__':
    net = LeNet()
    data = torch.rand((1, 3, 32, 32))
    res = net.forward(data)
    print(res)
