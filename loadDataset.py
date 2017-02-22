import numpy as np
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier


def readFromCSV(delimiter):
    filepaths = []
    with open("/Users/apple/PycharmProjects/Breast-Cancer-Classification/filepaths") as f:
        for line in f:
            filepaths.append(line.replace('\n', ''))
    train = genfromtxt(filepaths[0], delimiter=delimiter)
    test = genfromtxt(filepaths[1], delimiter=delimiter)
    dataset = [train, test]
    return dataset


