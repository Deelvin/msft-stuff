import typing

import numpy
import onnxruntime
import treelite_runtime

import utils.common


def sklearn_inference_runner(
    model: typing.Any, use_decision_function: bool = True
) -> typing.Callable:
    def classifier_inference(X: numpy.ndarray) -> typing.List[numpy.ndarray]:
        reference_output = [model.predict(X), model.predict_proba(X)]
        return reference_output

    def regressor_inference(X: numpy.ndarray) -> typing.List[numpy.ndarray]:
        reference_output = [
            model.predict(X),
        ]
        return reference_output

    def anomaly_detector_inference_ort_style(
        X: numpy.ndarray,
    ) -> typing.List[numpy.ndarray]:
        reference_output = [model.predict(X), model.decision_function(X)]
        return reference_output

    def anomaly_detector_inference_treelite_style(
        X: numpy.ndarray,
    ) -> typing.List[numpy.ndarray]:
        reference_output = [model.predict(X), -model.score_samples(X)]
        return reference_output

    if utils.common.is_sklearn_classifier(type(model)):
        return classifier_inference
    elif utils.common.is_sklearn_anomaly_detector(type(model)):
        if use_decision_function:
            return anomaly_detector_inference_ort_style
        else:
            return anomaly_detector_inference_treelite_style
    else:  # Regressor
        return regressor_inference


def ort_inference_runner(engine: onnxruntime.InferenceSession) -> typing.Callable:
    def inference_runner(
        input_dict: typing.Dict[str, numpy.ndarray]
    ) -> typing.List[numpy.ndarray]:
        return engine.run(None, input_dict)

    return inference_runner


def treelite_inference_runner(engine: treelite_runtime.Predictor) -> typing.Callable:
    def inference_runner(numpy_input: numpy.ndarray) -> typing.List[numpy.ndarray]:
        return [engine.predict(treelite_runtime.DMatrix(numpy_input), verbose=True)]

    return inference_runner
