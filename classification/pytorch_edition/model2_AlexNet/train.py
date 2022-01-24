import json
import os
import time

import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from model import AlexNet


def train(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ######################################## prepare data ########################################
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(size=(224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])}

    data_root = "/Users/cxl/data/"  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # get flower dataset path
    train_dir = os.path.join(image_path, "train")
    val_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)
    assert os.path.exists(val_dir), "{} path does not exist.".format(val_dir)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # get the classify data dict
    data_dict = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in data_dict.items())
    json_str = json.dumps(class_dict, indent=4)
    # write class dict into json file
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = 64
    # using multiprocessing to prepare train data
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    ########################################## start training ##########################################
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 10
    save_path = './myAlexNet.pth'
    best_accuracy = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train step
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        t1 = time.perf_counter()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch: [{} / {}], loss: {:.3f}".format(epoch + 1, epochs, loss)

        delta_t = time.perf_counter() - t1

        # validation
        """In eval mode, BN layer stops calculating and updating mean and std, 
           Dropout layer will allow all the active units to pass through"""
        model.eval()
        accuracy = 0.0
        with torch.no_grad():
            """stops gradient updates, saving GPU computing power and memory"""
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accuracy = accuracy / val_num

        print('[epoch %d], time: %d, train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, delta_t, running_loss / train_steps, val_accuracy))

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    myAlexNet = AlexNet(num_classes=5)
    train(model=myAlexNet)
