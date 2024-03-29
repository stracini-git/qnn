import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--wbits', type=int, default=32)
parser.add_argument('--abits', type=int, default=32)
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import time, uuid, pickle
import tensorflow.keras
import tensorflow as tf

import utils, ResNetBuilder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from localLayers import QuantizedDense
from tensorflow.keras import backend as KB

np.set_printoptions(edgeitems=3, linewidth=256)


def get_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.keras.backend.set_session(get_session())


def mnist():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    TestInput = mnist.test.images
    TestLabels = mnist.test.labels

    TrainInput = mnist.train.images
    TrainLabels = mnist.train.labels

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    return TrainInput, TrainLabels, TestInput, TestLabels, 10


def cifar10():
    from tensorflow.keras.datasets import cifar10

    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

    TrainInput = xtrain / 255.
    TestInput = xtest / 255.

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    TrainLabels = np.zeros((len(ytrain), 10))
    for i in range(0, len(ytrain)):
        TrainLabels[i, ytrain[i]] = 1

    TestLabels = np.zeros((len(ytest), 10))
    for i in range(0, len(ytest)):
        TestLabels[i, ytest[i]] = 1

    return np.ascontiguousarray(TrainInput), np.ascontiguousarray(TrainLabels), np.ascontiguousarray(TestInput), np.ascontiguousarray(TestLabels), 10


def training_schedule(schedule):
    schedule_epochs = schedule[0]
    schedule_lrates = schedule[1]

    expanded_lrates = []
    for i in range(len(schedule_epochs)):
        for j in range(schedule_epochs[i]):
            expanded_lrates.append(schedule_lrates[i])

    return expanded_lrates


def PrepareLeNet300(config):
    """
    Hardcoded LeNet-type of architecture,  could be moved somewhere else and generalized
    """

    initializer = config["initializer"]
    input_img = Input(shape=(28 * 28,))
    L300 = QuantizedDense(300, "relu", initializer, config)(input_img)
    L100 = QuantizedDense(100, "relu", initializer, config)(L300)
    L10 = QuantizedDense(10, "softmax", initializer, config)(L100)

    model = Model(input_img, L10)
    model._name = "LeNet300" + "_ID" + uuid.uuid4().hex[-7:]

    return model


def ResNetTrainer(network, data, mypath, config):
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        horizontal_flip=True,
        )

    Xtrain, Ytrain, Xtest, Ytest, nclasses = data
    datagen.fit(Xtrain)

    print("\nEvaluate network with no training:")
    TrainL0, TrainA0 = network.evaluate(Xtrain, Ytrain, batch_size=200, verbose=2)
    TestL0, TestA0 = network.evaluate(Xtest, Ytest, batch_size=200, verbose=2)

    TrainLoss = np.asarray([TrainL0])
    TrainAccuracy = np.asarray([TrainA0])

    TestLoss = np.asarray([TestL0])
    TestAccuracy = np.asarray([TestA0])

    maxtrainacc = TrainA0
    maxtestacc = TestA0

    loss, metric = network.metrics_names

    expanded_lrates = training_schedule(config["lr_schedule"])
    batchsize = config["batchsize"]
    maxepochs = len(expanded_lrates)

    epoch = 0
    while epoch < maxepochs:
        start_time = time.time()
        lr = expanded_lrates[epoch]
        KB.set_value(network.optimizer.lr, lr)

        fit_history = network.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=batchsize), validation_data=(Xtest, Ytest), epochs=1, verbose=2, shuffle=True)

        TrainLoss = np.append(TrainLoss, fit_history.history[loss])
        TestLoss = np.append(TestLoss, fit_history.history['val_loss'])

        TrainAccuracy = np.append(TrainAccuracy, fit_history.history[metric])
        TestAccuracy = np.append(TestAccuracy, fit_history.history['val_' + metric])

        if TestAccuracy[-1] > maxtestacc:
            predictions = network.predict(Xtest)
            np.save(mypath + "BestTestPredictions.npy", predictions)

        maxtrainacc = max(maxtrainacc, TrainAccuracy[-1])
        maxtestacc = max(maxtestacc, TestAccuracy[-1])

        print("epoch           - {}/{}".format(epoch + 1, maxepochs))
        print("wbits, abits    - {}, {} ".format(config["wbits"], config["abits"]))
        print("learn rate      - {:.13f}".format(lr))
        print("trn             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TrainLoss[-1], TrainAccuracy[-1], maxtrainacc, np.mean(TrainAccuracy[-5:])))
        print("tst             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TestLoss[-1], TestAccuracy[-1], maxtestacc, np.mean(TestAccuracy[-5:])))

        epoch += 1
        print("Execution time: {:.3f} seconds".format(time.time() - start_time))
        print("=" * 100, "\n")

    Logs = {"trainLoss": TrainLoss,
            "testLoss": TestLoss,
            "trainAccuracy": TrainAccuracy,
            "testAccuracy": TestAccuracy,
            }

    np.savetxt(mypath + 'TrainAccuracy.txt', TrainAccuracy, delimiter=',')
    np.savetxt(mypath + 'TestAccuracy.txt', TestAccuracy, delimiter=',')

    predictions = network.predict(Xtest)
    np.save(mypath + "TestPredictions.npy", predictions)

    file = open(mypath + "TrainLogs.pkl", "wb")
    pickle.dump(Logs, file)
    file.close()

    weights = utils.getkernels(network)
    utils.show_activations(network, Xtrain[0:1])

    file = open(mypath + "Weights.pkl", "wb")
    pickle.dump(weights, file)
    file.close()

    return Logs


def LeNetTrainer(network, data, mypath, config):
    Xtrain, Ytrain, Xtest, Ytest, nclasses = data

    print("\nEvaluate network with no training:")
    TrainL0, TrainA0 = network.evaluate(Xtrain, Ytrain, batch_size=200, verbose=2)
    TestL0, TestA0 = network.evaluate(Xtest, Ytest, batch_size=200, verbose=2)

    TrainLoss = np.asarray([TrainL0])
    TrainAccuracy = np.asarray([TrainA0])

    TestLoss = np.asarray([TestL0])
    TestAccuracy = np.asarray([TestA0])

    maxtrainacc = TrainA0
    maxtestacc = TestA0

    loss, metric = network.metrics_names

    expanded_lrates = training_schedule(config["lr_schedule"])
    batchsize = config["batchsize"]
    maxepochs = len(expanded_lrates)

    # custom train loop
    epoch = 0
    while epoch < maxepochs:
        start_time = time.time()
        lr = expanded_lrates[epoch]
        KB.set_value(network.optimizer.lr, lr)

        fit_history = network.fit(Xtrain, Ytrain, batch_size=batchsize, epochs=1, verbose=0, shuffle=True, validation_data=(Xtest, Ytest))

        TrainLoss = np.append(TrainLoss, fit_history.history[loss])
        TestLoss = np.append(TestLoss, fit_history.history['val_loss'])

        TrainAccuracy = np.append(TrainAccuracy, fit_history.history[metric])
        TestAccuracy = np.append(TestAccuracy, fit_history.history['val_' + metric])

        if TestAccuracy[-1] > maxtestacc:
            predictions = network.predict(Xtest)
            np.save(mypath + "BestTestPredictions.npy", predictions)

        maxtrainacc = max(maxtrainacc, TrainAccuracy[-1])
        maxtestacc = max(maxtestacc, TestAccuracy[-1])

        print("epoch           - {}/{}".format(epoch + 1, maxepochs))
        print("wbits, abits    - {}, {} ".format(config["wbits"], config["abits"]))
        print("learn rate      - {:.13f}".format(lr))
        print("trn             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TrainLoss[-1], TrainAccuracy[-1], maxtrainacc, np.mean(TrainAccuracy[-5:])))
        print("tst             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TestLoss[-1], TestAccuracy[-1], maxtestacc, np.mean(TestAccuracy[-5:])))

        epoch += 1
        print("Execution time: {:.3f} seconds".format(time.time() - start_time))
        print("=" * 100, "\n")

    Logs = {"trainLoss": TrainLoss,
            "testLoss": TestLoss,
            "trainAccuracy": TrainAccuracy,
            "testAccuracy": TestAccuracy,
            }

    np.savetxt(mypath + 'TrainAccuracy.txt', TrainAccuracy, delimiter=',')
    np.savetxt(mypath + 'TestAccuracy.txt', TestAccuracy, delimiter=',')

    predictions = network.predict(Xtest)
    np.save(mypath + "TestPredictions.npy", predictions)

    file = open(mypath + "TrainLogs.pkl", "wb")
    pickle.dump(Logs, file)
    file.close()

    weights = utils.getkernels(network)
    utils.show_activations(network, Xtrain[0:1])

    file = open(mypath + "Weights.pkl", "wb")
    pickle.dump(weights, file)
    file.close()

    return


def ResNet(config):
    # some particular config stuff goes in here
    config["batchsize"] = 64
    config["lr_schedule"] = [[80, 20, 20], [0.001, 0.0005, 0.0001]]

    version, n = 1, 3
    config["name"] = "ResNet_V" + str(version) + "_n" + str(n) + "_w" + str(config["wbits"]) + "_a" + str(config["abits"])
    mypath = utils.make_outputpath(config)
    file = open(mypath + "Config.pkl", "wb")
    pickle.dump(config, file)
    file.close()

    data = cifar10()
    network = ResNetBuilder.MakeResNet(data[0].shape[1:], version, n, config)
    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()
    print("All files will be available in:", mypath)

    ResNetTrainer(network, data, mypath, config)
    print("All files available in:", mypath)
    KB.clear_session()


def LeNet(config):
    # some particular config stuff goes in here
    config["batchsize"] = 64
    config["lr_schedule"] = [[15, 15, 40], [0.001, 0.0001, 0.00001]]

    if config["abits"] == 1:
        config["lr_schedule_lenet"] = [[15, 15, 40], [0.0002, 0.0001, 0.00001]]
        config["batchsize"] = 200

    config["name"] = "LeNet300_w" + str(config["wbits"]) + "_a" + str(config["abits"])
    mypath = utils.make_outputpath(config)
    file = open(mypath + "Config.pkl", "wb")
    pickle.dump(config, file)
    file.close()

    network = PrepareLeNet300(config)
    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()
    print("All files will be available in:", mypath)

    LeNetTrainer(network, mnist(), mypath, config)
    print("All files available in:", mypath)
    KB.clear_session()


def main(args):
    # some configuration stuff goes here
    config = {
        "basedir": "Outputs/" + utils.DateToString() + "/",
        "initializer": "normal",
        "activation": 'relu',
        "addbias": True,
        "wbits": args.wbits,
        "abits": args.abits
    }
    # copy all experiment files to the run folder, to make sure we get all
    utils.copyfilesto(config["basedir"])

    # overwrite default values here
    # config["wbits"] = 4
    # config["abits"] = 32

    LeNet(config)
    # ResNet(config)

    return


if __name__ == '__main__':
    main(args)
