from __future__ import print_function

import tensorflow as tf
import numpy as np
import uuid


def heconstant(p):
    def initializer(shape, dtype=None):
        nlp = np.prod(shape[:-1])
        a = np.sqrt(2 / nlp)
        distribution = np.random.choice([-1., 1.], shape, p=[p, 1 - p])
        return tf.Variable(a * distribution, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def magnitude(a):
    def initializer(shape, dtype=None):
        return tf.Variable(a * np.ones(shape), dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def activate(x, activationtype):
    if activationtype is None:
        return x

    if 'relu' == activationtype:
        return tf.keras.activations.relu(x)

    if 'softmax' in activationtype:
        return tf.keras.activations.softmax(x)

    if 'sigmoid' in activationtype:
        return tf.keras.activations.sigmoid(x)

    if 'swish' in activationtype:
        return tf.keras.activations.sigmoid(x) * x

    if "elu" in activationtype:
        return tf.keras.activations.elu(x)

    if "selu" in activationtype:
        return tf.keras.activations.selu(x)

    return x
