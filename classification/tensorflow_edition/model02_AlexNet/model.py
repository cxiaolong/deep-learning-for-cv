import tensorflow as tf


def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    """functional API edition"""
    input_image = tf.keras.layers.Input(shape=(im_width, im_height, 3), dtype=tf.float32)  # [None, 224, 224, 3]
    x = tf.keras.layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # [None, 227, 227, 3]
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')(x)  # [None, 55, 55, 96]
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)  # [None, 27, 27, 96]
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(x)  # [None, 27, 27, 256]
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)  # [None, 13, 13, 256]
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)  # [None, 13, 13, 384]
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)  # [None, 13, 13, 384]
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)  # [None, 13, 13, 256]
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)  # [None, 6, 6, 256]

    x = tf.keras.layers.Flatten()(x)  # [None, 6 * 6 * 256]
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(units=2048)(x)  # [None, 2048]
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(units=2048)(x)  # [None, 2048]
    x = tf.keras.layers.Dense(units=num_classes)(x)  # [None, num_classes]
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=outputs)
    return model


class AlexNet_v2(tf.keras.Model):
    """subclass edition"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(((1, 2), (1, 2))),
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=2048, activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=num_classes),
            tf.keras.layers.Softmax()
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs)
        x = self.flatten(x)
        outputs = self.classifier(x)
        return outputs


if __name__ == '__main__':
    input = tf.random.normal(shape=[1, 224, 224, 3])
    print(input)
    # my_alexnet1 = AlexNet_v1(num_classes=5)
    my_alexnet2 = AlexNet_v1(num_classes=5)
    # print(my_alexnet1(input))
    print(my_alexnet2(input))
