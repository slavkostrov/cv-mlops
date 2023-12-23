import numpy as np


def accuracy(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)
