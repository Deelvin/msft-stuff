import typing

import numpy
import sklearn.datasets


def get_classification_dataset() -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_classification(
        n_samples=10000,
        n_features=500,
        n_classes=100,
        n_informative=8,
        random_state=47,
        shuffle=False,
    )
    return X.astype("float32"), y


def get_regression_dataset() -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_regression(
        n_samples=10000, n_features=500, n_targets=1, random_state=47, shuffle=False
    )
    return X.astype("float32"), y
