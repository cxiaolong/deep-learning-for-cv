import json
import os
import time

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

from model import GoogLeNet


def train(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################## prepare data ########################################
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])}

    data_root = "/Users/cxl/data/"
    image_path = os.path.join(data_root, "flower_data")
    train_dir = os.path.join(image_path, "train")
    val_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "{} path does not exists".format(train_dir)
    assert os.path.exists(val_dir), "{} path does not exists".format(val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transform['val'])

    train_num, val_num = len(train_dataset), len(val_dataset)
    print("using {} images for training, {} images for validation".format(train_num, val_num))

    data_dict = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in data_dict.items())
    json_str = json.dumps(class_dict, indent=4)
    with open("class_indices.json", "w") as f:
        f.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ########################################## start training ##########################################
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 30
    best_accuracy = 0.0
    save_path = "./googlenet.pth"
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0

        t1 = time.perf_counter()
        for train_images, train_labels in train_loader:
            optimizer.zero_grad()
            logits, logits_aux2, logits_aux1 = model(train_images.to(device))
            # train step has three outputs: final output and two aux outputs
            loss0 = criterion(logits, train_labels.to(device))
            loss1 = criterion(logits_aux2, train_labels.to(device))
            loss2 = criterion(logits_aux1, train_labels.to(device))
            loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        delta_t = time.perf_counter() - t1

        # validation
        accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                outputs = model(val_images.to(device))
                # validation step only has one output
                y_pred = torch.max(outputs, dim=1)[1]
                accuracy += torch.eq(y_pred, val_labels.to(device)).sum().item()
            val_accuracy = accuracy / val_num

            print('[epoch %d], time: %d, train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, delta_t, running_loss / train_steps, val_accuracy))

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    train(net)