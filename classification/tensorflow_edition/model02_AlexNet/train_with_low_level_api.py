"""
This script using keras low level api to training
"""
import os

import tensorflow as tf

from model import AlexNet_v1, AlexNet_v2


def train_with_low_api(model):
    ########################################## prepare data ############################################
    data_root = "~/data/"  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # get flower dataset path
    train_dir = os.path.join(image_path, "train")
    val_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "can not find {}".format(train_dir)
    assert os.path.exists(val_dir), "can not find {}".format(val_dir)

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

    ################################# definite accuracy and loss ######################################
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    ################################# definite train and test step ######################################
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_obj(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_obj(y_true=labels, y_pred=predictions)

        val_loss(loss)
        val_accuracy(labels, predictions)

    ############################################ start training ###########################################
    for epoch in range(epochs):
        # clear history info
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for step in range(total_train // batch_size):
            train_images, train_labels = next(train_data_gen)
            train_step(train_images, train_labels)

        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)

        ######################################## print loss and accuracy ######################################
        template = 'Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            val_loss.result(),
            val_accuracy.result() * 100
        ))


if __name__ == '__main__':
    model1 = AlexNet_v1(num_classes=5)
    model2 = AlexNet_v2(num_classes=5)
    model2.build(input_shape=(64, 224, 224, 3))
    # train_with_low_api(model1)
    train_with_low_api(model2)
