import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # [None, 3, 224, 224] --> [None, 96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [None, 96, 55, 55] --> [None, 96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # [None, 96, 27, 27] --> [None, 256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [None, 256, 27, 27] --> [None, 256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # [None, 256, 27, 27] --> [None, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # [None, 384, 13, 13] --> [None, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [None, 384, 13, 13] --> [None, 256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [None, 256, 13, 13] --> [None, 256, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

        if init_weights:
            self._init_weights()

    def forward(self, inputs):
        x = self.features(inputs)
        x = torch.flatten(x, start_dim=1)
        outputs = self.classifier(x)
        return outputs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


# test model
if __name__ == '__main__':
    test_data = torch.rand(size=(1, 3, 224, 224))
    print(test_data)
    myAlexNet = AlexNet(num_classes=5, init_weights=True)
    out = myAlexNet(test_data)
    print(out)
