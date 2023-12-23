import numpy as np
from sklearn.metrics import f1_score


def f1(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)
