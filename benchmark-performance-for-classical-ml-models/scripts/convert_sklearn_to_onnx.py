import os
import pickle
import typing

import numpy
import onnxruntime
import skl2onnx
import tqdm

import utils.common
import utils.dataset


def is_classifier(model_type: typing.Type[typing.Any]) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.sklearn_classifier_list


def check_ort_inference(
    model_path: str,
    input_dict: typing.Dict[str, typing.List[numpy.ndarray]],
    reference_output: typing.List[numpy.ndarray],
) -> None:
    engine = onnxruntime.InferenceSession(model_path)
    with utils.common.Profiler("ONNX Runtime inference", show_latency=True):
        ort_output = engine.run(None, input_dict)

    assert len(reference_output) == len(ort_output)
    for reference, ort in zip(reference_output, ort_output):
        assert numpy.allclose(
            reference, ort.reshape(reference.shape), rtol=1.0e-5, atol=1.0e-5
        )


def convert_to_onnx_with_skl2onnx(
    model, model_name: str, input_name: str, input_shape: typing.Tuple
) -> str:
    save_root = os.path.join(utils.project_root(), "models", "skl2onnx")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, f"{model_name}.onnx")

    initial_type = [
        (
            input_name,
            skl2onnx.common.data_types.FloatTensorType(shape=input_shape),
        )
    ]
    onnx_model = skl2onnx.convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}} if is_classifier(type(model)) else None,
    )
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return save_path


def convert_models(
    target_models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    # Create dataset
    X, y = dataset_generator(n_samples=100)
    input_name = "input"
    input_shape = X.shape

    # Convert models
    for model_name in tqdm.tqdm(target_models):
        with open(
            os.path.join(
                utils.project_root(), "models", "sklearn", f"{model_name}.sklearn"
            ),
            "rb",
        ) as model_file:
            model = pickle.loads(model_file.read())

        model_path = convert_to_onnx_with_skl2onnx(
            model, model_name, input_name, input_shape
        )

        with utils.common.Profiler("Python sklearn inference", show_latency=True):
            reference_output = [
                model.predict(X).astype("float32"),
            ]
            if is_classifier(type(model)):
                reference_output.append(model.predict_proba(X))
        check_ort_inference(model_path, {input_name: X}, reference_output)


def main():
    convert_models(
        utils.common.sklearn_classifiers, utils.dataset.get_classification_dataset
    )
    convert_models(
        utils.common.sklearn_regressors, utils.dataset.get_regression_dataset
    )
    convert_models(utils.common.outlier_detectors, utils.dataset.get_regression_dataset)


if __name__ == "__main__":
    main()
