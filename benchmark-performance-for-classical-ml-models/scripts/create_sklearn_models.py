import os
import typing

import numpy
import sklearn.ensemble
import tqdm

import utils.common
import utils.dataset


@utils.common.SaveSKLearn(
    save_root=os.path.join(utils.project_root(), "models", "sklearn")
)
def create_sklearn_model(
    model_name: str, dataset: typing.Tuple[numpy.ndarray, numpy.ndarray]
) -> typing.Tuple[typing.Any, str]:
    save_name = f"{model_name}.sklearn"
    X, y = dataset

    model = getattr(sklearn.ensemble, model_name)(n_estimators=1000, random_state=47)
    model.fit(X, y)

    return model, save_name


def serialize_models(
    models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    X, y = dataset_generator(n_samples=100)
    for model_name in tqdm.tqdm(models):
        print("\n", model_name, "\n")
        create_sklearn_model(model_name, (X, y))


def main():
    serialize_models(
        utils.common.sklearn_classifiers, utils.dataset.get_classification_dataset
    )
    serialize_models(
        utils.common.sklearn_regressors, utils.dataset.get_regression_dataset
    )
    serialize_models(
        utils.common.outlier_detectors, utils.dataset.get_regression_dataset
    )


if __name__ == "__main__":
    main()
