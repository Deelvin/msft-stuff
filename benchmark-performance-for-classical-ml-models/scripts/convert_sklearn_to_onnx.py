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
import treelite
import treelite_runtime

import utils.common
import utils.dataset


def is_classifier(model_type: typing.Type[typing.Any]) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.sklearn_classifier_list


def is_anomaly_detector(model_type: typing.Type[typing.Any]) -> bool:
    import skl2onnx._supported_operators

    return model_type in skl2onnx._supported_operators.outlier_list


def get_sklearn_output(
    model: typing.Any, X: numpy.ndarray
) -> typing.List[numpy.ndarray]:
    with utils.common.Profiler("Python sklearn inference", show_latency=True):
        reference_output = [
            model.predict(X),
        ]
        if is_classifier(type(model)):
            reference_output.append(model.predict_proba(X))
        if is_anomaly_detector(type(model)):
            reference_output.append(model.decision_function(X))
    return reference_output


def get_ort_output(
    model_path: str, input_dict: typing.Dict[str, typing.List[numpy.ndarray]]
) -> typing.List[numpy.ndarray]:
    engine = onnxruntime.InferenceSession(model_path)
    with utils.common.Profiler("ONNX Runtime inference", show_latency=True):
        ort_output = engine.run(None, input_dict)
    return ort_output


def get_treelite_output(
    model_path: str, input_dict: typing.Dict[str, typing.List[numpy.ndarray]]
) -> typing.List[numpy.ndarray]:
    engine = treelite_runtime.Predictor(libpath=model_path)
    numpy_input = list(input_dict.values())[0]
    with utils.common.Profiler("Treelite inference", show_latency=True):
        input_data = treelite_runtime.DMatrix(numpy_input)
        treelite_output_probability = engine.predict(input_data, verbose=True)
        treelite_output = [treelite_output_probability]

    return treelite_output


@utils.common.onnx_saver(
    save_root=os.path.join(utils.project_root(), "models", "skl2onnx")
)
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


# TODO(agladyshev): call of hummingbird.ml._topology.convert.fix_graph should be commented
@utils.common.onnx_saver(
    save_root=os.path.join(utils.project_root(), "models", "hummingbird")
)
def convert_to_onnx_with_hummingbird(
    skl_model, model_name: str, input_dict: typing.Dict[str, numpy.ndarray]
) -> typing.Tuple[onnx.ModelProto, str]:
    save_name = f"hummingbird_{model_name}.onnx"

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


@utils.common.treelite_saver(
    save_root=os.path.join(utils.project_root(), "models", "treelite")
)
def convert_to_treelite(
    skl_model,
    model_name: str,
    _: typing.Dict[str, numpy.ndarray],
) -> typing.Tuple[treelite.frontend.Model, str]:
    save_name = f"treelite_{model_name}.so"
    model = treelite.sklearn.import_model(skl_model)
    return model, save_name


def convert_models(
    convert_function: typing.Callable[
        [typing.Any, str, typing.Dict[str, numpy.ndarray]], str
    ],
    target_models: typing.List[str],
    dataset_generator: typing.Callable,
) -> None:
    # Create dataset
    X, y = dataset_generator(n_samples=10000)
    input_dict = {"input": X}

    for model_name in tqdm.tqdm(target_models):
        # TODO(agladyshev):
        #   Out of memory for convert_to_onnx_with_hummingbird-GradientBoostingClassifier
        if (
            convert_function == convert_to_onnx_with_hummingbird
            and model_name == "GradientBoostingClassifier"
        ):
            continue
        # TODO(agladyshev): deadlock for skl2onnx-IsolationForest?
        if (
            convert_function == convert_to_onnx_with_skl2onnx
            and model_name == "IsolationForest"
        ):
            continue

        with open(
            os.path.join(
                utils.project_root(), "models", "sklearn", f"{model_name}.sklearn"
            ),
            "rb",
        ) as model_file:
            skl_model = pickle.loads(model_file.read())

        # Convert models
        model_path = convert_function(skl_model, model_name, input_dict)

        # Check inference output
        reference_output = get_sklearn_output(skl_model, X)
        engine_output = get_ort_output(model_path, input_dict)
        # engine_output = get_treelite_output(model_path, input_dict)
        assert len(reference_output) == len(engine_output)
        for reference, engine in zip(reference_output, engine_output):
            if "int" not in str(reference.dtype):
                numpy.testing.assert_allclose(
                    reference, engine.reshape(reference.shape), rtol=1.0e-4, atol=1.0e-4
                )


def main():
    for converter in [
        convert_to_onnx_with_skl2onnx,
        convert_to_onnx_with_hummingbird,
        convert_to_treelite,
    ]:
        convert_models(
            converter,
            utils.common.sklearn_classifiers,
            utils.dataset.get_classification_dataset,
        )
        convert_models(
            converter,
            utils.common.sklearn_regressors,
            utils.dataset.get_regression_dataset,
        )
        convert_models(
            converter,
            utils.common.outlier_detectors,
            utils.dataset.get_regression_dataset,
        )


if __name__ == "__main__":
    main()
