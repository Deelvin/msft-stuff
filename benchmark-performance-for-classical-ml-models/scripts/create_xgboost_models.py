import os
import typing

import numpy
import tqdm
import xgboost

import utils.common
import utils.dataset
import utils.save


@utils.save.xgboost_saver(
    save_root=os.path.join(utils.project_root(), "models", "xgboost")
)
def create_xgboost_model(
    model_name: str, dataset: typing.Tuple[numpy.ndarray, numpy.ndarray]
) -> typing.Tuple[typing.Any, str]:
    save_name = f"{model_name}.xgboost"
    kwargs = {}
    if "Rank" in model_name:
        kwargs["qid"] = utils.dataset.get_qid(dataset[0].shape[0])
    model = getattr(xgboost, model_name)(n_estimators=1000, n_jobs=16, random_state=47)
    model.fit(*dataset, **kwargs)

    return model, save_name


def serialize_models(
    models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    X, y = dataset_generator(n_samples=200)
    for model_name in tqdm.tqdm(models):
        print("\n", model_name, "\n")
        create_xgboost_model(model_name, (X, y))


def main():
    serialize_models(
        utils.common.xgboost_classifiers, utils.dataset.get_classification_dataset
    )
    serialize_models(
        utils.common.xgboost_regressors, utils.dataset.get_regression_dataset
    )
    serialize_models(
        utils.common.xgboost_rankers, utils.dataset.get_classification_dataset
    )


if __name__ == "__main__":
    main()
