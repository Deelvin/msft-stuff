import typing

import numpy
import sklearn.datasets


def get_classification_dataset() -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_classification(
        n_samples=10,
        n_features=224 * 224 * 3,
        n_classes=1000,
        n_informative=11,
        random_state=47,
        shuffle=False,
    )
    return X.astype("float32"), y


def get_regression_dataset() -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_regression(
        n_samples=10, n_features=100000, n_targets=1000, random_state=47, shuffle=False
    )
    return X.astype("float32"), y
