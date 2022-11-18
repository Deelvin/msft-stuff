import os
import pickle
import typing

import sklearn.ensemble
import tqdm

import utils.common
import utils.dataset


def serialize_models(
    models: typing.List[str], dataset_generator: typing.Callable
) -> None:
    # Create save dir
    save_root = os.path.join(utils.project_root(), "models", "sklearn")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Fit model
    X, y = dataset_generator()
    for model_name in tqdm.tqdm(models):
        print("\n", model_name, "\n")
        model = getattr(sklearn.ensemble, model_name)(random_state=47)
        model.fit(X, y)

        # Serialize and save model
        model_path = os.path.join(save_root, f"{model_name}.sklearn")
        with open(model_path, "wb") as model_file:
            serialized_model = pickle.dumps(model)
            model_file.write(serialized_model)


def main():
    serialize_models(
        utils.common.sklearn_classifiers, utils.dataset.get_classification_dataset
    )
    serialize_models(
        utils.common.sklearn_regressors, utils.dataset.get_regression_dataset
    )


if __name__ == "__main__":
    main()
