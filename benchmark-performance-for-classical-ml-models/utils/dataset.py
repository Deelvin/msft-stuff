import typing

import numpy
import sklearn.datasets


def get_classification_dataset() -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    return X, y
