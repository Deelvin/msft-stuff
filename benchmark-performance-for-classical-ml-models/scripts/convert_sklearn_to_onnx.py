import abc
import os
import typing

import numpy
import onnxruntime
import skl2onnx
import sklearn.ensemble
import tqdm

import utils.dataset


def is_classifier(model_type: abc.ABCMeta) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.sklearn_classifier_list


def check_ort_inference(
    model_path: str,
    input_dict: typing.Dict[str, numpy.ndarray],
    reference_output: typing.List,
) -> None:
    engine = onnxruntime.InferenceSession(model_path)
    with utils.Profiler("ONNX Runtime inference", show_latency=True):
        ort_output = engine.run(None, input_dict)

    assert len(reference_output) == len(ort_output)
    for reference, ort in zip(reference_output, ort_output):
        assert numpy.allclose(reference, ort)


def convert_to_onnx(
    model, save_path: str, input_name: str, input_shape: typing.Tuple
) -> None:
    initial_type = [
        (
            input_name,
            skl2onnx.common.data_types.FloatTensorType(shape=[1, *input_shape]),
        )
    ]
    onnx_model = skl2onnx.convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}} if is_classifier(type(model)) else None,
    )
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def convert_models(
    target_models: typing.List[str], X: numpy.ndarray, y: numpy.ndarray
) -> None:
    reference_input = X[0]
    input_name = "input"
    input_shape = reference_input.shape

    for model_name in tqdm.tqdm(target_models):
        model = getattr(sklearn.ensemble, model_name)()
        model.fit(X, y)

        model_path = os.path.join(
            utils.project_root(), "models", "onnx", f"{model_name}.onnx"
        )
        convert_to_onnx(model, model_path, input_name, input_shape)

        with utils.Profiler("Python sklearn inference", show_latency=True):
            reference_output = [
                model.predict([reference_input]),
                model.predict_proba([reference_input]),
            ]
        check_ort_inference(
            model_path, {input_name: [reference_input]}, reference_output
        )


def main():
    target_classifiers = [
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
    ]
    convert_models(target_classifiers, *utils.dataset.get_classification_dataset())

    target_regressors = [
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
    ]
    # convert_models(target_regressors, *utils.dataset.get_regression_dataset())


if __name__ == "__main__":
    main()
