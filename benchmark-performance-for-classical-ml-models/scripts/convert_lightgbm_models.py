import os
import tempfile
import typing

import hummingbird.ml
import numpy.testing
import onnx
import tqdm
import treelite

import utils.common
import utils.dataset
import utils.inference
import utils.load
import utils.save


def get_lightgbm_output(
    model_path: str,
    input_dict: typing.Dict[str, numpy.ndarray],
) -> typing.List[numpy.ndarray]:
    engine = utils.load.load_lightgbm(model_path)
    runner = utils.inference.lightgbm_inference_runner(engine)

    with utils.common.Profiler("Python lightgbm inference", show_latency=True):
        reference_output = runner(*input_dict.values())
    return reference_output


# TODO(agladyshev): call of hummingbird.ml._topology.convert.fix_graph should be commented
@utils.save.onnx_saver(
    save_root=os.path.join(utils.project_root(), "models", "hummingbird_lightgbm")
)
def convert_to_onnx_with_hummingbird(
    lgbm_model_path: str, model_name: str, input_dict: typing.Dict[str, numpy.ndarray]
) -> typing.Tuple[onnx.ModelProto, str]:
    save_name = f"hummingbird_{model_name}.onnx"

    xgb_model = utils.load.load_xgboost(lgbm_model_path)

    input_names = list(input_dict.keys())
    onnx_model = hummingbird.ml.convert(
        xgb_model,
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
    save_root=os.path.join(utils.project_root(), "models", "treelite_lightgbm")
)
def convert_to_treelite(
    lgbm_model_path,
    model_name: str,
    _: typing.Dict[str, numpy.ndarray],
) -> typing.Tuple[treelite.frontend.Model, str]:
    save_name = f"treelite_{model_name}.so"

    # TODO(agladyshev): dirty hack
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_path = os.path.join(temp_dir, f"{model_name}.dump")
        lgbm_model = utils.load.load_lightgbm(lgbm_model_path)
        lgbm_model.booster_.save_model(dump_path)
        model = treelite.frontend.Model.load(dump_path, model_format="lightgbm")
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
        lgbm_model_path = os.path.join(
            utils.project_root(), "models", "lightgbm", f"{model_name}.lightgbm"
        )

        # Convert models
        model_path = convert_function(lgbm_model_path, model_name, input_dict)

        # Check inference output
        reference_output = get_lightgbm_output(lgbm_model_path, input_dict)
        engine_output = inference_function(model_path, input_dict)

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
        (convert_to_onnx_with_hummingbird, utils.common.get_ort_output),
        (convert_to_treelite, utils.common.get_treelite_output),
    ]:
        convert_models(
            utils.common.lightgbm_classifiers,
            convert_function,
            utils.dataset.get_classification_dataset,
            inference_function,
        )
        convert_models(
            utils.common.lightgbm_regressors,
            convert_function,
            utils.dataset.get_regression_dataset,
            inference_function,
        )
        convert_models(
            utils.common.lightgbm_rankers,
            convert_function,
            utils.dataset.get_classification_dataset,
            inference_function,
        )


if __name__ == "__main__":
    main()
