import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from model import AlexNet_v1, AlexNet_v2


def train(model):
    ################################# prepare data ######################################
    data_root = "/Users/cxl/data/"  # get data root path
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
    im_width = 224
    batch_size = 64
    epochs = 10

    # create data generator with data augmentation
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.,
                                                                            horizontal_flip=True)
    val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               target_size=(im_height, im_width),
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               class_mode='categorical')
    val_data_gen = val_image_generator.flow_from_directory(directory=val_dir,
                                                           target_size=(im_height, im_width),
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           class_mode='categorical')
    total_train = train_data_gen.n
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train, total_val))

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform  key and value of the class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())

    # write inverse_dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    ##################################### start training #######################################
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlexNet.h5',
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          monitor='val_loss')
    history = model.fit(x=train_data_gen,
                        epochs=epochs,
                        callbacks=[model_checkpoint],
                        validation_data=val_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        validation_steps=total_val // batch_size)

    ################################### plot loss and accuracy ##################################
    history_dict = history.history
    train_loss = history_dict.get('loss')
    train_accuracy = history_dict.get('accuracy')
    val_loss = history_dict.get('val_loss')
    val_accuracy = history_dict.get('val_accuracy')

    # figure loss
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure accuracy
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    model1 = AlexNet_v1(num_classes=5)
    model2 = AlexNet_v2(num_classes=5)
    model2.build(input_shape=(64, 224, 224, 3))
    train(model2)
