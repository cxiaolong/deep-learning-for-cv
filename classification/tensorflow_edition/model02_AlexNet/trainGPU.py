import matplotlib.pyplot as plt
import tensorflow as tf
import os, json

from model import AlexNet_v1, AlexNet_v2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_with_gpu(model):
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    ################################# prepare data ######################################
    data_root = "~/data/"  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # get flower dataset path
    train_dir = os.path.join(image_path, "train/")
    val_dir = os.path.join(image_path, "val/")
    assert os.path.exists(train_dir), "can not find {}".format(train_dir)
    assert os.path.exists(val_dir), "can not find {}".format(val_dir)

    # create directory for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # definite train params
    im_height = 224
    im_weight = 224
    batch_size = 64
    epochs = 10