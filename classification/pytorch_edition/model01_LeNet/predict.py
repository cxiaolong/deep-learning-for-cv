import torch
from PIL import Image
from torchvision.transforms import transforms

from model import LeNet


def predict():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('LeNet.pth'))

    image = Image.open('1.png')
    image = transform(image)  # [C, H, W]
    image = torch.unsqueeze(image, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(image)
        print(outputs)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
        print(classes[int(predict)])


if __name__ == '__main__':
    predict()
