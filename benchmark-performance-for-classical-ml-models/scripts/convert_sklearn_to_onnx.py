import os
import pickle
import typing

import numpy
import numpy.testing
import onnx
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
        if "int" not in str(reference.dtype):
            numpy.testing.assert_allclose(
                reference, ort.reshape(reference.shape), rtol=1.0e-4, atol=1.0e-4
            )


class SaveONNX:
    def __init__(self, save_root: str):
        self.save_root = save_root

    def __call__(
        self,
        convert_func: typing.Callable[..., typing.Tuple[onnx.ModelProto, str]],
    ):
        def wrapper(*args, **kwargs):
            if not os.path.exists(self.save_root):
                os.makedirs(self.save_root)

            onnx_model, save_name = convert_func(*args, **kwargs)

            save_path = os.path.join(self.save_root, save_name)
            with open(save_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            return save_path

        return wrapper


@SaveONNX(save_root=os.path.join(utils.project_root(), "models", "skl2onnx"))
def convert_to_onnx_with_skl2onnx(
    model, model_name: str, input_name: str, input_shape: typing.Tuple
) -> typing.Tuple[onnx.ModelProto, str]:
    save_name = f"skl2onnx_{model_name}.onnx"

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

    return onnx_model, save_name


def convert_to_onnx_with_hummingbird():
    pass


def convert_models(
    target_models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    # Create dataset
    X, y = dataset_generator(n_samples=10000)
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
                model.predict(X),
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
    # TODO(agladyshev): deadlock?
    # convert_models(utils.common.outlier_detectors, utils.dataset.get_regression_dataset)


if __name__ == "__main__":
    main()
