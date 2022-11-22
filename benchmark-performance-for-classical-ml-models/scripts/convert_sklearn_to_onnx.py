import os
import pickle
import typing

import hummingbird.ml
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
    skl_model, model_name: str, input_dict: typing.Dict[str, numpy.ndarray]
) -> typing.Tuple[onnx.ModelProto, str]:
    def get_tensor_type(tensor: numpy.ndarray):
        if tensor.dtype == numpy.float32:
            tensor_type = skl2onnx.common.data_types.FloatTensorType(shape=tensor.shape)
        else:
            raise NotImplementedError(f"Type {tensor.dtype} not supported.")
        return tensor_type

    save_name = f"skl2onnx_{model_name}.onnx"

    initial_type = [
        (
            input_name,
            get_tensor_type(input_dict[input_name]),
        )
        for input_name in input_dict.keys()
    ]
    onnx_model = skl2onnx.convert_sklearn(
        skl_model,
        initial_types=initial_type,
        options={id(skl_model): {"zipmap": False}}
        if is_classifier(type(skl_model))
        else None,
    )

    return onnx_model, save_name


@SaveONNX(save_root=os.path.join(utils.project_root(), "models", "hummingbird"))
def convert_to_onnx_with_hummingbird(
    skl_model, model_name: str, input_dict: typing.Dict[str, numpy.ndarray]
) -> typing.Tuple[onnx.ModelProto, str]:
    save_name = f"hummingbird_{model_name}.onnx"

    # TODO(agladyshev): call of hummingbird.ml._topology.convert.fix_graph should be commented
    input_names = list(input_dict.keys())
    onnx_model = hummingbird.ml.convert(
        skl_model,
        "onnx",
        test_input=[input_dict[input_name] for input_name in input_names]
        if len(input_names) > 1
        else input_dict[input_names[0]],
        extra_config=dict(
            input_names=input_names,
        ),
    ).model

    return onnx_model, save_name


def convert_models(
    target_models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    # Create dataset
    X, y = dataset_generator(n_samples=10000)
    input_dict = {"input": X}

    # Convert models
    for model_name in tqdm.tqdm(target_models):
        with open(
            os.path.join(
                utils.project_root(), "models", "sklearn", f"{model_name}.sklearn"
            ),
            "rb",
        ) as model_file:
            model = pickle.loads(model_file.read())

        model_path = convert_to_onnx_with_skl2onnx(model, model_name, input_dict)
        model_path = convert_to_onnx_with_hummingbird(model, model_name, input_dict)

        with utils.common.Profiler("Python sklearn inference", show_latency=True):
            reference_output = [
                model.predict(X),
            ]
            if is_classifier(type(model)):
                reference_output.append(model.predict_proba(X))
        check_ort_inference(model_path, input_dict, reference_output)


def main():
    convert_models(
        utils.common.sklearn_classifiers, utils.dataset.get_classification_dataset
    )
    convert_models(
        utils.common.sklearn_regressors, utils.dataset.get_regression_dataset
    )
    # TODO(agladyshev): deadlock for skl2onnx?
    # convert_models(utils.common.outlier_detectors, utils.dataset.get_regression_dataset)


if __name__ == "__main__":
    main()
