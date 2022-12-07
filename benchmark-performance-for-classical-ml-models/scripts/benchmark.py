import argparse
import csv
import os
import typing

import tqdm

import utils
import utils.benchmark_lightgbm
import utils.benchmark_onnx
import utils.benchmark_sklearn
import utils.benchmark_treelite
import utils.benchmark_xgboost
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
                utils.benchmark_treelite.TreeliteModel,
            ),
            (
                os.path.join(models_root, "sklearn", f"{model_name}.sklearn"),
                get_dataset_generator(model_name),
                utils.benchmark_sklearn.SklearnModel,
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
                utils.benchmark_treelite.TreeliteModel,
            ),
            (
                os.path.join(models_root, "xgboost", f"{model_name}.xgboost"),
                get_dataset_generator(model_name),
                utils.benchmark_xgboost.XGBoostModel,
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
                utils.benchmark_treelite.TreeliteModel,
            ),
            (
                os.path.join(models_root, "lightgbm", f"{model_name}.lightgbm"),
                get_dataset_generator(model_name),
                utils.benchmark_lightgbm.LightgbmModel,
            ),
        ]
    else:
        raise ValueError(f"Unsupported model {model_name}")


def benchmark_meta():
    return [(model_name, benchmark_row(model_name)) for model_name in list_of_models()]


def return_value_or_none(f: typing.Callable) -> typing.Callable:
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(e)
            return None

    return wrapper


def make_table_header() -> typing.List[str]:
    return [
        "Model name",
        "skl2onnx (ms)",
        "Treelite (ms)",
        "Hummingbird (ms)",
        "Python (ms)",
    ]


def make_table_row(performance_results: typing.List) -> typing.List[str]:
    csv_none = "-"

    def replace_dot_with_comma(value):
        return str(value).replace(".", ",") if value is not None else value

    def replace_if_none(value):
        return csv_none if value is None else value

    return [
        replace_if_none(replace_dot_with_comma(value)) for value in performance_results
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shift",
        required=False,
        default=0,
    )
    args = parser.parse_args()

    benchmark_file_path = os.path.join(utils.project_root(), "benchmark_results.csv")
    with open(benchmark_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(make_table_header())

    models_meta = benchmark_meta()[args.shift :]
    for model_name, meta in tqdm.tqdm(models_meta):
        model_results = []
        if not is_sklearn_model(model_name):
            # Fill skl2onnx
            model_results.append(None)
        for model_path, dataset_generator, benchmark_factory in meta:
            if "hummingbird" in model_path and "LGBMClassifier" in model_path:
                # Need ~120GB of RAM
                model_results.append(None)
                continue

            X, _ = dataset_generator(10000)
            input_dict = {"input": X}

            benchmark_factory = return_value_or_none(benchmark_factory)
            benchmarker = benchmark_factory(model_path, input_dict)

            if benchmarker:
                benchmark_function = benchmarker.benchmark
                mean, std = benchmark_function(num_trials=5, num_runs_per_trial=10)
                model_results.append(mean)
            else:
                model_results.append(None)

        with open(benchmark_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerow([model_name, *make_table_row(model_results)])


if __name__ == "__main__":
    main()
