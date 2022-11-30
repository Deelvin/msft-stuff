import os
import typing

import lightgbm
import numpy
import tqdm

import utils.common
import utils.dataset
import utils.save


@utils.save.lightgbm_saver(
    save_root=os.path.join(utils.project_root(), "models", "lightgbm")
)
def create_lightgbm_model(
    model_name: str, dataset: typing.Tuple[numpy.ndarray, numpy.ndarray]
) -> typing.Tuple[typing.Any, str]:
    save_name = f"{model_name}.lightgbm"
    model_kwargs = {}
    fit_kwargs = {}
    if "Rank" in model_name:
        model_kwargs["label_gain"] = numpy.arange(0, numpy.max(dataset[1]) + 1)
        fit_kwargs["group"] = numpy.unique(
            utils.dataset.get_qid(dataset[0].shape[0]), return_counts=True
        )[1]
    model = getattr(lightgbm, model_name)(
        n_estimators=1000, n_jobs=16, random_state=47, **model_kwargs
    )
    model.fit(*dataset, **fit_kwargs)

    return model, save_name


def serialize_models(
    models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    X, y = dataset_generator(n_samples=200)
    for model_name in tqdm.tqdm(models):
        print("\n", model_name, "\n")
        create_lightgbm_model(model_name, (X, y))


def main():
    serialize_models(
        utils.common.lightgbm_classifiers, utils.dataset.get_classification_dataset
    )
    serialize_models(
        utils.common.lightgbm_regressors, utils.dataset.get_regression_dataset
    )
    serialize_models(
        utils.common.lightgbm_rankers, utils.dataset.get_classification_dataset
    )


if __name__ == "__main__":
    main()
