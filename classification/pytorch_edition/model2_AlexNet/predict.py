import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model import AlexNet


def predict(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # load image
    img_path = "../../test_images/test1.jpeg"
    assert os.path.exists(img_path), "file: '{}' does not exists".format(img_path)

    img = Image.open(img_path)
    plt.imshow(img)
    # to [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # read class indict
    with open("class_indices.json", "r") as f:
        class_indict = json.load(f)

    model.to(device)
    # load model weight
    weight_path = "./myAlexNet.pth"
    assert os.path.exists(weight_path), "file: '{}' does not exists".format(weight_path)
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).numpy()

        print("class: {}   prob: {:.3}".format(class_indict[str(predict_class)], predict[predict_class].numpy()))


if __name__ == '__main__':
    model = AlexNet(num_classes=5)
    predict(model)