from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# supress annoying TF messages at the beginning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as KB
import numpy as np
from localFunctions import heconstant, activate, magnitude
import QFunctions


def KernelInitializer(initializer, p1):
    if initializer == 'normal':
        ki = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

    if initializer == 'glorot':
        ki = tf.compat.v1.keras.initializers.glorot_normal()

    if initializer == 'he':
        ki = tf.compat.v1.keras.initializers.he_normal()

    if initializer == "heconstant":
        ki = heconstant(p1)

    return ki


def quantize_activations(nbits):
    if nbits == 1:
        return QFunctions.qrelu1

    if nbits == 2:
        return QFunctions.qrelu2

    if nbits == 4:
        return QFunctions.qrelu4

    if nbits == 8:
        return QFunctions.qrelu8

    if nbits == 16:
        return QFunctions.qrelu16

    if nbits == 32:
        return QFunctions.qrelu32

    return tf.identity


def quantize_weights(nbits):
    if nbits == 1:
        return QFunctions.q1

    if nbits == 2:
        return QFunctions.q2

    if nbits == 4:
        return QFunctions.q4

    if nbits == 8:
        return QFunctions.q8

    if nbits == 16:
        return QFunctions.q16

    if nbits == 32:
        return QFunctions.q32

    return tf.identity


def get_kernel_biases(name, kernel, bias):
    k = KB.eval(kernel)
    if bias is not None:
        b = KB.eval(bias)

    print("Layer {}".format(name))
    print("  total number of weights: {:7d} | unique: {:7d}".format(k.size, np.unique(k).size))
    print("  krnl: min | max | mean | std:    {:.13f} | {:.13f} | {:.13f} | {:.13f}".format(np.min(k), np.max(k), np.mean(k), np.std(k)))
    if bias is not None:
        print("  total number of biases:  {:7d} | unique: {:7d}".format(b.size, np.unique(b).size))
        print("  bias: min | max | mean | std:    {:.13f} | {:.13f} | {:.13f} | {:.13f}".format(np.min(b), np.max(b), np.mean(b), np.std(b)))

    return k


class QuantizedConv2D(Layer):
    def __init__(self, filters, ksize, activation, initializer, stride, config, **kwargs):
        self.filters = filters
        self.ksize = ksize
        self.stride = stride
        self.initializer = initializer
        self.addbias = config["addbias"]
        self.wbits = config["wbits"]
        self.abits = config["abits"]

        self.activation = activation

        # quantization functions
        self.wqnt = quantize_weights(self.wbits)
        self.aqnt = quantize_activations(self.abits)

        if stride is not None:
            self.stride = stride

        super(QuantizedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        bias_shape = (self.filters,)
        krnl_shape = list((self.ksize, self.ksize)) + [input_shape.as_list()[-1], self.filters]

        kernel_initializer = KernelInitializer(self.initializer, 0.5)

        if self.addbias:
            a = np.sqrt(2 / np.prod(bias_shape[:-1]))
            ashape = (1,)
            self.x_bias = self.add_weight(name='x_bias', shape=bias_shape, initializer=kernel_initializer, trainable=True)
            self.s_bias = self.add_weight(name='s_bias', shape=ashape, initializer=magnitude(a), trainable=True)
            self.bias = tf.abs(self.s_bias) * self.wqnt(self.x_bias)

        a = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        ashape = (1,)
        self.x_weights = self.add_weight(name='x_weights', shape=krnl_shape, initializer=kernel_initializer, trainable=True)
        self.a_weights = self.add_weight(name='s_weights', shape=ashape, initializer=magnitude(a), trainable=True)
        self.kernel = tf.abs(self.a_weights) * self.wqnt(self.x_weights)

        super(QuantizedConv2D, self).build(input_shape)

    def call(self, x):
        y = KB.conv2d(x, self.kernel, strides=(self.stride, self.stride), padding='same')
        if self.addbias:
            y += self.bias

        if self.activation is not None:
            if self.abits != 0:
                y = tf.clip_by_value(y, 0, (2 ** self.abits) - 1)

            return self.aqnt(y)
        else:
            return y

    def compute_output_shape(self, input_shape):
        return (input_shape.as_list()[1], self.output_dim)

    def get_weights(self):
        return super(QuantizedConv2D, self).get_weights()

    def set_weights(self, weights):
        super(QuantizedConv2D, self).set_weights(weights)

    def get_kernel(self):
        return get_kernel_biases(self.name, self.kernel, self.bias)


class QuantizedDense(Layer):

    def __init__(self, output_dim, activation, initializer, config, **kwargs):
        self.output_dim = output_dim
        self.initializer = initializer
        self.addbias = config["addbias"]
        self.wbits = config["wbits"]
        self.abits = config["abits"]

        self.activation = activation

        # quantization functions
        self.wqnt = quantize_weights(self.wbits)
        self.aqnt = quantize_activations(self.abits)

        super(QuantizedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        bias_shape = (self.output_dim,)
        krnl_shape = (input_shape.as_list()[1], self.output_dim)

        kernel_initializer = KernelInitializer(self.initializer, 0.5)

        if self.addbias:
            a = np.sqrt(2 / np.prod(bias_shape[:-1]))
            ashape = (1,)
            self.x_bias = self.add_weight(name='x_bias', shape=bias_shape, initializer=kernel_initializer, trainable=True)
            self.a_bias = self.add_weight(name='a_bias', shape=ashape, initializer=magnitude(a), trainable=True)
            self.bias = tf.abs(self.a_bias) * self.wqnt(self.x_bias)

        a = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        ashape = (1,)
        self.x_weights = self.add_weight(name='x_weights', shape=krnl_shape, initializer=kernel_initializer, trainable=True)
        self.a_weights = self.add_weight(name='a_weights', shape=ashape, initializer=magnitude(a), trainable=True)
        self.kernel = tf.abs(self.a_weights) * self.wqnt(self.x_weights)

        super(QuantizedDense, self).build(input_shape)

    def call(self, x):
        y = KB.dot(x, self.kernel)
        if self.addbias:
            y += self.bias

        if self.activation != "softmax":
            y = tf.clip_by_value(y, 0, (2 ** self.abits) - 1)
            act = self.aqnt(y)
        else:
            act = activate(y, self.activation)

        return act

    def compute_output_shape(self, input_shape):
        return input_shape.as_list()[1], self.output_dim

    def get_weights(self):
        return super(QuantizedDense, self).get_weights()

    def set_weights(self, weights):
        super(QuantizedDense, self).set_weights(weights)

    def get_kernel(self):
        return get_kernel_biases(self.name, self.kernel, self.bias)
