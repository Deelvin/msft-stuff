import typing

import numpy
import sklearn.datasets


def get_classification_dataset(
    n_samples: int,
) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=500,
        n_classes=100,
        n_informative=8,
        random_state=47,
        shuffle=False,
    )
    return X.astype("float32"), y


def get_regression_dataset(
    n_samples: int,
) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    X, y = sklearn.datasets.make_regression(
        n_samples=n_samples, n_features=500, n_targets=1, random_state=47, shuffle=False
    )
    return X.astype("float32"), y


def get_qid(n_samples: int) -> numpy.ndarray:
    numpy.random.seed(47)
    return numpy.sort(
        numpy.random.randint(low=1, high=int(n_samples / 3), size=n_samples)
    )
