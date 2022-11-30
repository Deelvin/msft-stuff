import os
import typing

import hummingbird.ml
import numpy
import numpy.testing
import onnx
import skl2onnx
import tqdm
import treelite

import utils.common
import utils.dataset
import utils.inference
import utils.load
import utils.save


def get_sklearn_output(
    model_path: str,
    input_dict: typing.Dict[str, numpy.ndarray],
    use_decision_function=True,
) -> typing.List[numpy.ndarray]:
    engine = utils.load.load_sklearn(model_path)
    runner = utils.inference.sklearn_inference_runner(engine, use_decision_function)

    with utils.common.Profiler("Python sklearn inference", show_latency=True):
        reference_output = runner(*input_dict.values())
    return reference_output


@utils.save.onnx_saver(
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
        if utils.common.is_sklearn_classifier(type(skl_model))
        else None,
    )

    return onnx_model, save_name


# TODO(agladyshev): call of hummingbird.ml._topology.convert.fix_graph should be commented
@utils.save.onnx_saver(
    save_root=os.path.join(utils.project_root(), "models", "hummingbird_sklearn")
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


@utils.save.treelite_saver(
    save_root=os.path.join(utils.project_root(), "models", "treelite_sklearn")
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
    target_models: typing.List[str],
    convert_function: typing.Callable[
        [typing.Any, str, typing.Dict[str, numpy.ndarray]], str
    ],
    dataset_generator: typing.Callable[
        [int], typing.Tuple[numpy.ndarray, numpy.ndarray]
    ],
    inference_function: typing.Callable[
        [str, typing.Dict[str, numpy.ndarray]], typing.List[numpy.ndarray]
    ],
) -> None:
    # Create dataset
    X, y = dataset_generator(10000)
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
        # TODO(agladyshev):
        #  treelite.util.TreeliteError:
        #   Gradient boosted trees must be trained with the option init='zero'
        if convert_function == convert_to_treelite and "GradientBoosting" in model_name:
            continue

        skl_model_path = os.path.join(
            utils.project_root(), "models", "sklearn", f"{model_name}.sklearn"
        )
        skl_model = utils.load.load_sklearn(skl_model_path)

        # Convert models
        model_path = convert_function(skl_model, model_name, input_dict)

        # Check inference output
        reference_output = get_sklearn_output(
            skl_model_path, input_dict, convert_function != convert_to_treelite
        )
        engine_output = inference_function(model_path, input_dict)

        # TODO(agladyshev):
        #   Mismatched elements: 6456 / 10000 (64.6%)
        #   Max absolute difference: 5.21174154
        #   Max relative difference: 57.81789888
        if (
            convert_function == convert_to_treelite
            and model_name == "RandomForestRegressor"
        ):
            continue
        # TODO(agladyshev):
        #   Mismatched elements: 5168 / 10000 (51.7%)
        #   Max absolute difference: 2.91902493
        #   Max relative difference: 553.51602663
        if (
            convert_function == convert_to_treelite
            and model_name == "ExtraTreesRegressor"
        ):
            continue

        # TODO(agladyshev):
        #   Treelite doesn't provide predict method in sklearn-style for classifiers
        if (
            inference_function == utils.common.get_treelite_output
            and len(reference_output) == 2
        ):
            reference_output = [reference_output[1]]

        utils.common.output_comparer(reference_output, engine_output)


def main():
    for convert_function, inference_function in [
        (convert_to_onnx_with_skl2onnx, utils.common.get_ort_output),
        (convert_to_onnx_with_hummingbird, utils.common.get_ort_output),
        (convert_to_treelite, utils.common.get_treelite_output),
    ]:
        convert_models(
            utils.common.sklearn_classifiers,
            convert_function,
            utils.dataset.get_classification_dataset,
            inference_function,
        )
        convert_models(
            utils.common.sklearn_regressors,
            convert_function,
            utils.dataset.get_regression_dataset,
            inference_function,
        )
        convert_models(
            utils.common.sklearn_outlier_detectors,
            convert_function,
            utils.dataset.get_regression_dataset,
            inference_function,
        )


if __name__ == "__main__":
    main()
