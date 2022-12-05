import os
import typing

import utils.benchmark_onnx
import utils.common
import utils.dataset


def is_sklearn_model(model_name: str) -> bool:
    return (
        model_name in utils.common.sklearn_classifiers
        or model_name in utils.common.sklearn_regressors
        or model_name in utils.common.sklearn_outlier_detectors
    )


def is_xgboost_model(model_name: str) -> bool:
    return (
        model_name in utils.common.xgboost_classifiers
        or model_name in utils.common.xgboost_regressors
        or model_name in utils.common.xgboost_rankers
    )


def is_lightgbm_model(model_name: str) -> bool:
    return (
        model_name in utils.common.lightgbm_classifiers
        or model_name in utils.common.lightgbm_regressors
        or model_name in utils.common.lightgbm_rankers
    )


def is_classifier(model_name: str) -> bool:
    return (
        model_name in utils.common.sklearn_classifiers
        or model_name in utils.common.xgboost_classifiers
        or model_name in utils.common.lightgbm_classifiers
    )


def is_regressor(model_name: str) -> bool:
    return (
        model_name in utils.common.sklearn_regressors
        or model_name in utils.common.xgboost_regressors
        or model_name in utils.common.lightgbm_regressors
    )


def is_ranker(model_name: str) -> bool:
    return (
        model_name in utils.common.xgboost_rankers
        or model_name in utils.common.lightgbm_rankers
    )


def is_outlier_detector(model_name: str) -> bool:
    return model_name in utils.common.sklearn_outlier_detectors


def list_of_models() -> typing.List[str]:
    sklearn_models = [
        model_name
        for sublist in [
            [regressor, classifier]
            for classifier, regressor in zip(
                utils.common.sklearn_classifiers, utils.common.sklearn_regressors
            )
        ]
        for model_name in sublist
    ] + utils.common.sklearn_outlier_detectors
    xgboost_models = (
        [*utils.common.xgboost_classifiers]
        + [*utils.common.xgboost_regressors]
        + [*utils.common.xgboost_rankers]
    )
    lightgbm_models = (
        [*utils.common.lightgbm_classifiers]
        + [*utils.common.lightgbm_regressors]
        + [*utils.common.lightgbm_rankers]
    )
    return [*sklearn_models] + [*xgboost_models] + [*lightgbm_models]


def get_dataset_generator(model_name: str) -> typing.Callable:
    if is_classifier(model_name) or is_ranker(model_name):
        return utils.dataset.get_classification_dataset
    elif is_regressor(model_name) or is_outlier_detector(model_name):
        return utils.dataset.get_regression_dataset
    else:
        raise ValueError(f"Unsupported model {model_name}")


def benchmark_row(model_name: str) -> typing.List[typing.Tuple]:
    models_root = os.path.join(utils.project_root(), "models")
    if is_sklearn_model(model_name):
        return [
            (
                os.path.join(models_root, "skl2onnx", f"skl2onnx_{model_name}.onnx"),
                get_dataset_generator(model_name),
                utils.benchmark_onnx.OnnxModel,
            ),
            (
                os.path.join(
                    models_root, "hummingbird_sklearn", f"hummingbird_{model_name}.onnx"
                ),
                get_dataset_generator(model_name),
                utils.benchmark_onnx.OnnxModel,
            ),
            (
                os.path.join(
                    models_root, "treelite_sklearn", f"treelite_{model_name}.so"
                ),
                get_dataset_generator(model_name),
                None,
            ),
            (
                os.path.join(models_root, "sklearn", f"{model_name}.sklearn"),
                get_dataset_generator(model_name),
                None,
            ),
        ]
    elif is_xgboost_model(model_name):
        return [
            (
                os.path.join(
                    models_root, "hummingbird_xgboost", f"hummingbird_{model_name}.onnx"
                ),
                get_dataset_generator(model_name),
                utils.benchmark_onnx.OnnxModel,
            ),
            (
                os.path.join(
                    models_root, "treelite_xgboost", f"treelite_{model_name}.so"
                ),
                get_dataset_generator(model_name),
                None,
            ),
            (
                os.path.join(models_root, "xgboost", f"{model_name}.xgboost"),
                get_dataset_generator(model_name),
                None,
            ),
        ]
    elif is_lightgbm_model(model_name):
        return [
            (
                os.path.join(
                    models_root,
                    "hummingbird_lightgbm",
                    f"hummingbird_{model_name}.onnx",
                ),
                get_dataset_generator(model_name),
                utils.benchmark_onnx.OnnxModel,
            ),
            (
                os.path.join(
                    models_root, "treelite_lightgbm", f"treelite_{model_name}.so"
                ),
                get_dataset_generator(model_name),
                None,
            ),
            (
                os.path.join(models_root, "lightgbm", f"{model_name}.lightgbm"),
                get_dataset_generator(model_name),
                None,
            ),
        ]
    else:
        raise ValueError(f"Unsupported model {model_name}")


def benchmark_meta():
    return [(model_name, benchmark_row(model_name)) for model_name in list_of_models()]


def main():
    for model_name, meta in benchmark_meta():
        for model_path, dataset_generator, benchmark_factory in meta:
            X, _ = dataset_generator(10)
            input_dict = {"input": X}
            benchmarker = benchmark_factory(model_path, input_dict)

            mean, std = benchmarker.benchmark(num_trials=10, num_runs_per_trial=10)


if __name__ == "__main__":
    main()
