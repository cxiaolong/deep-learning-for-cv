import json
import os

import torch
from PIL import Image
from torchvision import transforms

from model import GoogLeNet


def predict(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image_path = "../../test_images/test1.jpeg"
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with open("class_indices.json", "r") as json_str:
        class_indict = json.load(json_str)

    model.to(device)
    weight_path = "./googlenet.pth"
    assert os.path.exists(weight_path), "{} does not exists".format(weight_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(f=weight_path, map_location=device), strict=False)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        pred = torch.softmax(output, dim=0)
        pred_class = torch.argmax(pred).numpy()

    print("class: {}   prob: {:.3}".format(class_indict[str(pred_class)], pred[pred_class].numpy()))


if __name__ == '__main__':
    model = GoogLeNet(num_classes=5, aux_logits=False)
    predict(model)
