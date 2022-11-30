import time
import typing

import numpy

sklearn_classifiers = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
]


sklearn_regressors = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
]

sklearn_outlier_detectors = [
    "IsolationForest",
]


xgboost_classifiers = [
    "XGBClassifier",
]


xgboost_regressors = [
    "XGBRegressor",
]


xgboost_rankers = [
    "XGBRanker",
]


class Profiler:
    def __init__(self, name="", show_latency=False, iterations_number=1):
        self.name = name
        self.show_latency = show_latency
        self.iterations_number = iterations_number

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type, value, traceback):
        end_time = time.perf_counter()
        if self.show_latency:
            print(
                "[{}] elapsed time: {:.3f} ms ({} iterations)".format(
                    self.name,
                    (end_time - self.start_time) * 1000 / self.iterations_number,
                    self.iterations_number,
                )
            )


def output_comparer(reference_output, engine_output) -> None:
    assert len(reference_output) == len(engine_output)
    for reference, engine in zip(reference_output, engine_output):
        if "int" not in str(reference.dtype):
            numpy.testing.assert_allclose(
                reference, engine.reshape(reference.shape), rtol=1.0e-2, atol=1.0e-2
            )


def is_sklearn_classifier(model_type: typing.Type[typing.Any]) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.sklearn_classifier_list


def is_sklearn_anomaly_detector(model_type: typing.Type[typing.Any]) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.outlier_list
