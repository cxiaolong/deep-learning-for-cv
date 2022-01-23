import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import AlexNet_v1, AlexNet_v2


def predict(model):
    im_height = 224
    im_width = 224

    # load and resize image
    img_path = '../../test_images/test1.jpeg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # rescale pixel and append dims
    img = np.array(img) / 255.
    img = np.expand_dims(img, axis=0)

    # read class indict
    json_path = "class_indices.json"
    with open(json_path, 'r') as json_file:
        class_indict = json.load(json_file)

    # prediction
    result = np.squeeze(model.predict(img))
    idx = np.argmax(result)
    print("class: {},    probability: {}".format(class_indict[str(idx)], result[idx]))


if __name__ == '__main__':
    model1 = AlexNet_v1(num_classes=5)
    # model2 = AlexNet_v2(num_classes=5)
    # model2.build(input_shape=(64, 224, 224, 3))

    # load weight
    weight_path = "./save_weights/myAlexNet.h5"
    assert os.path.exists(weight_path), "file: '{}' does not exists".format(weight_path)
    model1.load_weights(filepath=weight_path)

    predict(model1)
