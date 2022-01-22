import matplotlib.pyplot as plt
import tensorflow as tf
import os, json, glob

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
    data_root = "/Users/cxl/data/"  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # get flower dataset path
    train_dir = os.path.join(image_path, "train/")
    val_dir = os.path.join(image_path, "val/")
    assert os.path.exists(train_dir), "can not find {}".format(train_dir)
    assert os.path.exists(val_dir), "can not find {}".format(val_dir)

    # create directory for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # create class dict
    data_class = [cl for cl in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cl))]
    class_num = len(data_class)
    class_dict = dict((val, index) for index, val in enumerate(data_class))

    # reverse key and value of class_dict
    inverse_dict = dict((val, key) for key, val in class_dict.items())
    # write inverse_dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # read train and validation data
    train_image_list = glob.glob(pathname=train_dir+"/*/*.jpg")
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]
    train_num = len(train_image_list)
    val_image_list = glob.glob(pathname=val_dir+"/*/*.jpg")
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]
    val_num = len(val_image_list)

    # definite train params
    im_height = 224
    im_width = 224
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005

    def process_path(img_path, label):
        """"""
        label = tf.one_hot(indices=label, depth=class_num)
        image = tf.io.read_file(filename=img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                                 .map(map_func=process_path, num_parallel_calls=AUTOTUNE)\
                                 .repeat()\
                                 .batch(batch_size=batch_size)\
                                 .prefetch(buffer_size=AUTOTUNE)
    # load validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(map_func=process_path, num_parallel_calls=AUTOTUNE).repeat().batch(batch_size)

    #################################### start training ######################################
    model.summary()
    # high level api learning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=tf.keras.metrics.CategoricalAccuracy())
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlexNet.h5',
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          monitor='val_loss')
    history = model.fit(x=train_dataset,
                        epochs=epochs,
                        callbacks=[model_checkpoint],
                        validation_data=val_dataset,
                        steps_per_epoch=train_num // batch_size,
                        validation_steps=val_num // batch_size)

    ################################### plot loss and accuracy ##################################
    history_dict = history.history
    train_loss = history_dict.get('loss')
    train_accuracy = history_dict.get('categorical_accuracy')
    val_loss = history_dict.get('val_loss')
    val_accuracy = history_dict.get('val_categorical_accuracy')

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
    train_with_gpu(model1)
    # train_with_gpu(model2)