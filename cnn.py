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


def batch_norm(momentum=0.9):
    return BatchNormalization(momentum=momentum)


def dc_d(norm=None):
    def add_block(filters):
        model.add(Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=W_INIT))
        if norm == "batch":
            model.add(batch_norm())
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.2))

    model = keras.Sequential()
    # [n, 128, 128, 3]
    # if use_bn:
    #     model.add(batch_norm())
    # model.add(GaussianNoise(0.02))
    add_block(32)
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


def dc_g(input_shape, norm="batch"):
    if norm == "instance":
        _norm = InstanceNormalization
    else:
        _norm = batch_norm
    block = lambda filters: [
        UpSampling2D((2, 2), interpolation="bilinear"),
        Conv2D(filters, 4, 1, "same", kernel_initializer=W_INIT),
        _norm(), ReLU(),
    ]
    return keras.Sequential([
        # [n, latent]
        Dense(4 * 4 * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        _norm(), ReLU(),
        Reshape((4, 4, 128)),
        # 4
        *block(128),
        # 8
        *block(126),
        # 16
        *block(64),
        # 32
        *block(64),
        # 64
        *block(32),
        # 128
        Conv2D(3, 7, 1, padding="same", kernel_initializer=W_INIT),
        _norm(),
        Activation(keras.activations.tanh),
    ])


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, bottlenecks=2, norm=None):
        super().__init__()
        self.bs = keras.Sequential(
            [ResBottleneck(filters, norm) for _ in range(bottlenecks)]
        )

    def call(self, x, training=None):
        o = self.bs(x, training=training)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters, norm=None):
        super().__init__()
        self.norm = norm
        c = filters // 3
        self.b = keras.Sequential([Conv2D(c, 1, 1, "same", kernel_initializer=W_INIT)])
        self._add_norm(self.b)
        self.b.add(LeakyReLU(0.2))
        self.b.add(Conv2D(filters, 4, 1, "same", kernel_initializer=W_INIT))
        self._add_norm(self.b)

    def _add_norm(self, b):
        if self.norm == "batch":
            b.add(batch_norm())
        elif self.norm == "instance":
            b.add(InstanceNormalization())

    def call(self, x, training=None):
        b = self.b(x, training=training)
        x = tf.nn.leaky_relu(b + x, alpha=0.2)
        return x


def resnet_g(input_shape, img_shape=(128, 128, 3), norm="instance"):
    h, w = img_shape[0], img_shape[1]
    _h, _w = 4, 4
    m = keras.Sequential([
        keras.Input(input_shape),
        Dense(_h * _w * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        Reshape((_h, _w, 128)),
    ], name="resnet_g")
    if norm == "instance":
        m.add(InstanceNormalization())
    elif norm == "batch":
        m.add(batch_norm())
    c = _c = 128
    while True:
        strides = [1, 1]
        if _h < h:
            _h *= 2
            strides[0] = 2
        if _w < w:
            _w *= 2
            strides[1] = 2
        m.add(UpSampling2D(strides, interpolation="bilinear"))
        if c != _c:
            c = _c
            m.add(Conv2D(_c, 1, 1, "same"))
        m.add(ResBlock(filters=c, bottlenecks=1, norm=norm))
        if _w >= w and _h >= h:
            break
        c = max(int(c / 2), 64)

    # m.add(Conv2D(64, 5, 1, "same"))
    # if norm == "instance":
    #     m.add(InstanceNormalization())
    # elif norm == "batch":
    #     m.add(batch_norm())
    # m.add(LeakyReLU(0.2))
    m.add(Conv2D(3, 7, 1, "same", activation="tanh"))
    return m


def resnet_d(input_shape=(128, 128, 3), norm=None):
    if norm == "batch":
        norm = None
    _h, _w = input_shape[0], input_shape[1]
    h, w = 8, 8
    m = keras.Sequential([
        # Conv2D(64, 7, 1, "same", kernel_initializer=W_INIT),
    ], name="resnet_d")
    if norm == "instance":
        m.add(InstanceNormalization())
        m.add(LeakyReLU(0.2))
    c = 16
    while True:
        strides = [1, 1]
        if _h > h:
            _h //= 2
            strides[0] = 2
        if _w > w:
            _w //= 2
            strides[1] = 2
        m.add(Conv2D(c, 3, strides, "same", kernel_initializer=W_INIT))
        m.add(ResBlock(filters=c, bottlenecks=2, norm=norm))
        c = min(int(2 * c), 128)
        if _w <= w and _h <= h:
            break
    # m.add(Conv2D(256, 3, 2, "same"))    # 4^4
    # if norm == "instance":
    #     m.add(InstanceNormalization())
    # m.add(LeakyReLU(0.2))
    # m.add(Flatten())
    return m
