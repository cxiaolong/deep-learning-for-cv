"""
训练脚本有参考
tensorflow官方教程：https://tensorflow.google.cn/tutorials/quickstart/advanced
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ssl

from model import LeNet

ssl._create_default_https_context = ssl._create_unverified_context


def prepare_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.

    # # 查看数据
    # imgs = x_train[:3]
    # labs = y_train[:3]
    # print(labs)
    # plot_imgs = np.hstack(imgs)
    # plt.imshow(plot_imgs, cmap='gray')
    # plt.show()

    # 将数据从三维-->四维
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')

    # 数据集打乱及分批次
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    return train_ds, test_ds


@tf.function
def train_step(train_data, model, loss_obj, optimizer, train_loss, train_accuracy):
    train_images, train_labels = train_data
    with tf.GradientTape() as tape:
        predictions = model(train_images, training=True)
        loss = loss_obj(train_labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(train_labels, predictions)


def test_step(test_data, model, loss_obj, test_loss, test_accuracy):
    test_images, test_labels = test_data
    predictions = model(test_images, training=False)
    loss = loss_obj(test_labels, predictions)

    test_loss(loss)
    test_accuracy(test_labels, predictions)


def train(model, train_ds, test_ds):
    # 训练轮次
    epochs = 5
    # 为训练选择优化器、损失函数、评估指标（损失值、准确率）
    optimizer = tf.keras.optimizers.Adam()
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 迭代训练
    for epoch in range(epochs):
        # 清楚状态
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_state()
        test_accuracy.reset_states()

        for train_data in train_ds:
            train_step(train_data, model, loss_obj, optimizer, train_loss, train_accuracy)
        for test_data in test_ds:
            test_step(test_data, model, loss_obj, test_loss, test_accuracy)

        print(
            f'Epoch: {epoch + 1},'
            f'Loss: {train_loss.result()},'
            f'Accuracy: {train_accuracy.result() * 100},'
            f'Test Loss: {test_loss.result()},'
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )


if __name__ == '__main__':
    model = LeNet()
    train_ds, test_ds = prepare_data()

    train(model, train_ds, test_ds)