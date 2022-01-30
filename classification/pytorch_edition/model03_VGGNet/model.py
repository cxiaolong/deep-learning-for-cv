import torch
import torch.nn as nn

# official pre-trained model weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, inputs):
        x = self.features(inputs)  # [N, 3, 224, 224]  --> [N, 512, 7, 7]
        x = torch.flatten(x, start_dim=1)  # [N, 512, 7, 7]  --> [N, 512 * 7 * 7]
        outputs = self.classifier(x)  # [N, 512 * 7 * 7]  --> [N, num_classes]
        return outputs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.append(maxpool2d)
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers.append(conv2d)
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: {} not in config dict!".format(model_name)

    cfg = cfgs[model_name]
    model = VGG(features=make_features(cfg), **kwargs)
    return model


if __name__ == '__main__':
    test_data = torch.rand(size=(1, 3, 224, 224))
    model = vgg("vgg16", num_classes=5, init_weights=True)
    out = model(test_data)
    print(out)
