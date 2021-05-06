import os
import uuid
from datetime import datetime
from shutil import copy2

import numpy as np
from tensorflow.keras import backend as KB


def DateToString():
    now = datetime.now()
    datename = str(now)
    return datename.split(" ")[0].replace("-", ".")


def get_kernel(net):
    K = []

    for l in range(1, len(net.layers)):
        if "quantized" not in net.layers[l].name:
            continue

        k = net.layers[l].get_kernel()
        K.append(k)

    return K


def copyfilesto(mypath):
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    copy2("localFunctions.py", mypath)
    copy2("localLayers.py", mypath)
    copy2("QFunctions.py", mypath)
    copy2("ResNetBuilder.py", mypath)
    copy2("Trainer.py", mypath)
    copy2("utils.py", mypath)


def make_outputpath(config):
    RunID = uuid.uuid4().hex

    basedir = config["basedir"]
    mypath = basedir + config['name']
    mypath += "/" + config['initializer']
    mypath += "/" + RunID[-7:] + "/"

    copyfilesto(mypath)

    return mypath


def show_activations(network, inputimg):
    x = inputimg

    inp = network.input  # input placeholder
    # outputs = [layer.output for layer in network.layers if 'batch_normalization' not in layer.name][1:]  # all layer outputs except first (input) layer and batchnorm
    outputs = [layer.output for layer in network.layers][1:]  # all layer outputs except first (input) layer
    names = [layer.name for layer in network.layers][1:]  # all layer outputs except first (input) layer
    functor = KB.function(inp, outputs)  # evaluation function

    # Testing
    layer_outputvalues = functor([x])

    activations = []
    for layername, layerout in zip(names, layer_outputvalues):
        l = layerout
        print("layer:", layername, " act min/max/mean/std/uniq", np.min(l), np.max(l), np.mean(l), np.std(l), np.unique(l).size)
        activations.append(l)


def getkernels(net):
    weights = []

    for l in range(1, len(net.layers)):
        if "quantized" in net.layers[l].name:
            w = net.layers[l].get_kernel()
            weights.append(w)

    return weights
