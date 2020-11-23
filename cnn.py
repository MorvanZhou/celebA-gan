from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf

W_INIT = keras.initializers.RandomNormal(0, 0.02)


class InstanceNormalization_(keras.layers.Layer):
    """Batch Instance Normalization Layer (https://arxiv.org/abs/1805.07925)."""

    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):

        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) * (tf.math.rsqrt(ins_sigma + self.epsilon))

        return x_ins * self.gamma + self.beta

try:
    from tensorflow_addons.layers import InstanceNormalization
except ImportError:
    InstanceNormalization = InstanceNormalization_


def dc_d(norm="batch"):
    def add_block(filters):
        model.add(Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=W_INIT))
        if norm == "batch":
            model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

    model = keras.Sequential()
    # [n, 128, 128, 3]
    # if use_bn:
    #     model.add(BatchNormalization())
    model.add(GaussianNoise(0.02))
    add_block(16)
    # 64
    add_block(32)
    # 32
    add_block(64)
    # 16
    add_block(128)
    # 8
    add_block(256)
    # 4
    model.add(Flatten())
    return model


def dc_g(input_shape):
    block = lambda filters: [
        UpSampling2D((2, 2)),
        Conv2D(filters, 4, 1, "same", kernel_initializer=W_INIT),
        BatchNormalization(), ReLU()
    ]
    return keras.Sequential([
        # [n, latent]
        Dense(4 * 4 * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(), ReLU(),
        Reshape((4, 4, 128)),
        # 4
        *block(64),
        # 8
        *block(64),
        # 16
        *block(64),
        # 32
        *block(32),
        # 64
        *block(16),
        # 128
        Conv2D(3, 7, 1, padding="same", kernel_initializer=W_INIT),
        BatchNormalization(),
        Activation(keras.activations.tanh),
    ])


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, activation=None, bottlenecks=2, norm=None, inner_relu=True):
        super().__init__()
        self.activation = activation
        self.bn = keras.Sequential(
            [ResBottleneck(filters, norm, inner_relu=inner_relu)]
        )
        if bottlenecks > 1:
            self.bn.add(ReLU())
            for _ in range(1, bottlenecks):
                self.bn.add(ResBottleneck(filters, norm, inner_relu=inner_relu))

    def call(self, x, training=None):
        o = self.bn(x, training=training)
        if self.activation is not None:
            o = self.activation(o)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters, norm=None, inner_relu=True):
        super().__init__()
        c = filters // 4
        self.b = keras.Sequential([Conv2D(c, 1, strides=1, padding="same", kernel_initializer=W_INIT)])
        if norm == "batch":
            self.b.add(BatchNormalization())
        elif norm == "instance":
            self.b.add(InstanceNormalization())
        self.b.add(ReLU() if inner_relu else LeakyReLU())
        self.b.add(Conv2D(filters, 4, strides=1, padding="same", kernel_initializer=W_INIT))
        if norm == "batch":
            self.norm = BatchNormalization()
        elif norm == "instance":
            self.norm = InstanceNormalization()
        else:
            self.norm = None

    def call(self, x, training=None):
        b = self.b(x, training=training)
        x = b + x
        if self.norm is not None:
            x = self.norm(x)
        return x


def resnet_g(input_shape, img_shape=(128, 128, 3), norm="instance"):
    h, w = img_shape[0], img_shape[1]
    _h, _w = 4, 4
    m = keras.Sequential([
        keras.Input(input_shape),
        Dense(_h * _w * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(),
        Reshape((_h, _w, 128)),
    ], name="resnet_g")
    c = _c = 128
    while True:
        strides = [1, 1]
        if _h < h:
            _h *= 2
            strides[0] = 2
        if _w < w:
            _w *= 2
            strides[1] = 2
        m.add(UpSampling2D(strides))
        if c != _c:
            c = _c
            m.add(Conv2D(_c, 1, 1, "same"))
            if norm == "instance":
                m.add(InstanceNormalization())
        m.add(ResBlock(filters=c, bottlenecks=1, norm=norm))
        if _w >= w and _h >= h:
            break
        _c = max(int(c / 2), 128)

    m.add(ResBlock(128, bottlenecks=1, norm=norm, inner_relu=True))
    m.add(ResBlock(128, bottlenecks=1, norm=norm, inner_relu=True))
    m.add(Conv2D(3, 5, 1, "same"))
    m.add(Activation(keras.activations.tanh))
    return m


def resnet_d(input_shape=(128,128,3), norm="instance"):
    _h, _w = input_shape[0], input_shape[1]
    h, w = 4, 4
    m = keras.Sequential(name="resnet_d")
    c = 32
    while True:
        strides = [1, 1]
        if _h > h:
            _h //= 2
            strides[0] = 2
        if _w > w:
            _w //= 2
            strides[1] = 2
        m.add(Conv2D(c, 4, strides, "same", kernel_initializer=W_INIT))
        if norm == "batch":
            m.add(BatchNormalization())
        elif norm == "instance":
            m.add(InstanceNormalization())
        m.add(ResBlock(filters=c, bottlenecks=1, norm=norm, inner_relu=False))
        c = min(int(2 * c), 128)
        if _w <= w and _h <= h:
            break
    m.add(Flatten())
    return m
