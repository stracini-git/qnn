import glob, os
import pickle
import sys

import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import random
import socket

if socket.gethostname() != "CLJ-C-000CQ" and socket.gethostname() != "kneon":
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findfiles(mypath, fname):
    files = []

    for path in Path(mypath).rglob(fname):
        # print(path)
        files.append(path)
        # print(files[-1])

    return files


def plot_quantizationmatrix(mypath, title=""):
    QA, QW = getqaqw(mypath)

    Na = len(sorted(np.unique(np.asarray(QA))))
    Nw = len(sorted(np.unique(np.asarray(QW))))
    abits = sorted(np.unique(np.asarray(QA)))
    wbits = sorted(np.unique(np.asarray(QW)))

    qamap = {}
    for i in range(Na):
        qamap[abits[i]] = i

    qwmap = {}
    for i in range(Nw):
        qwmap[wbits[i]] = i

    Logs = findfiles(mypath, 'TrainLogs.pkl')
    Logs = sorted(Logs)
    print("working in: {}, found {} files".format(mypath, len(Logs)))
    if len(Logs) == 0:
        print("no necessary files found in path")
        return

    mean_accuracies = np.zeros((Na, Nw))
    entries = np.zeros_like(mean_accuracies)

    fig, axes = plt.subplots(1, 1, figsize=(15, 18), dpi=60, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=30)

    for l in Logs:
        fname = l.as_posix()
        LogFile = pickle.load(open(l, "rb"))

        config = pickle.load(open(fname[:-13] + "Config.pkl", "rb"))
        qa = int(config["abits"])
        qw = int(config["wbits"])

        testAccuracy = np.mean(LogFile['testAccuracy'][-1])
        mean_accuracies[qamap[qa], qwmap[qw]] += testAccuracy
        entries[qamap[qa], qwmap[qw]] += 1

    # divide by the number of runs we have for each quantization (activation, weight) pair
    mean_accuracies[entries != 0] /= entries[entries != 0]

    # if there are empty slots then fill them with the average of the whole matrix
    accuracies = np.zeros((Na, Nw))
    accuracies[mean_accuracies != 0] = mean_accuracies[mean_accuracies != 0]
    accuracies[mean_accuracies == 0] = np.mean(mean_accuracies[mean_accuracies != 0])
    axes.imshow(accuracies)

    for lin in range(Na):
        for col in range(Nw):
            all = "{:<.2f}".format(100 * accuracies[lin, col])
            fontsize = 25
            color = "white" if accuracies[lin, col] < np.mean(accuracies) - 2 * np.std(accuracies) else "black"
            if lin == col:
                fontsize = 30
            axes.annotate(all, xy=(col, lin), horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color=color)

    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(1))

    axes.set_title("Accuracy", fontsize=25)
    axes.set_xticklabels([''] + wbits, fontsize=23)
    axes.set_yticklabels([''] + abits, fontsize=23)
    axes.set_xlabel("Weights bit-depth", fontsize=25)
    axes.set_ylabel("Activations bit-depth", fontsize=25)

    plt.tight_layout(pad=1)
    plt.show()
    return


def getqaqw(mypath):
    """
    finds the different values for quantization bits for weights and activations
    """
    Logs = findfiles(mypath, 'TrainLogs.pkl')
    QA = []
    QW = []

    for l in Logs:
        fname = l.as_posix()
        print(fname)
        qa = int(fname.split("/")[2].split("_")[-1][1:])
        qw = int(fname.split("/")[2].split("_")[-2][1:])
        QA.append(qa)
        QW.append(qw)

    return QA, QW


def main():
    plot_quantizationmatrix("Outputs/2021.04.29_test/", "LeNet300, MNIST")
    return


if __name__ == '__main__':
    main()
