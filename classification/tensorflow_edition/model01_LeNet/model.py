import tensorflow as tf


class LeNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding='valid', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = self.fc3(x)
        return outputs


# test
if __name__ == '__main__':
    net = LeNet()
    data = tf.random.normal((1, 32, 32, 3))
    res = net(data)
    print(res)