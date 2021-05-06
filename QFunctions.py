import tensorflow as tf


def ggrad(dy):
    return dy


def qrelu(x, bits):
    n = (2 ** bits) - 1

    denom = tf.math.reduce_max(x)
    y0_1 = tf.keras.activations.relu(tf.math.ceil(n * x / denom) / n)
    y = denom * y0_1

    return y


@tf.custom_gradient
def qrelu1(x):
    bits = 1
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def qrelu2(x):
    bits = 2
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def qrelu4(x):
    bits = 4
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def qrelu8(x):
    bits = 8
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def qrelu16(x):
    bits = 16
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def qrelu32(x):
    bits = 32
    y = qrelu(x, bits)

    def grad(dy):
        return dy

    return y, grad


def ladder(x, bits):
    n = 2 ** (bits - 1)
    denom = tf.math.reduce_max(tf.abs(x))
    y = tf.math.round(n * x / denom) / n

    return y


def broken_ladder(x, bits):
    n = 2 ** (bits - 1)
    denom = tf.reduce_max(tf.abs(x))
    y1 = tf.keras.activations.relu(tf.ceil(n * x / denom) / n)
    y2 = tf.keras.activations.relu(-tf.floor(n * x / denom) / n) * tf.sign(x)
    y = y1 + y2

    return y


def quantization(x, bits):
    return broken_ladder(x, bits)
    # return ladder(x, bits)


@tf.custom_gradient
def q1(x):
    y = quantization(x, 1)

    return y, ggrad


@tf.custom_gradient
def q2(x):
    y = quantization(x, 2)

    return y, ggrad


@tf.custom_gradient
def q4(x):
    y = quantization(x, 4)
    return y, ggrad


@tf.custom_gradient
def q8(x):
    y = quantization(x, 8)

    return y, ggrad


@tf.custom_gradient
def q16(x):
    y = quantization(x, 16)

    return y, ggrad


@tf.custom_gradient
def q32(x):
    y = quantization(x, 32)

    return y, ggrad
