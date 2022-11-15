import os

import skl2onnx
import sklearn.ensemble
import tqdm

import utils.dataset


def convert_to_onnx(classifier, save_path: str) -> None:
    print(type(classifier))
    initial_type = [
        ("float_input", skl2onnx.common.data_types.FloatTensorType([None, 4]))
    ]
    onnx_model = skl2onnx.convert_sklearn(classifier, initial_types=initial_type)
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    target_classifiers = [
        "RandomForestClassifier",
    ]

    X, y = utils.dataset.get_classification_dataset()

    for classifier_name in tqdm.tqdm(target_classifiers):
        classifier = getattr(sklearn.ensemble, classifier_name)()
        classifier.fit(X, y)
        convert_to_onnx(
            classifier, os.path.join(".", "models", "onnx", f"{classifier_name}.onnx")
        )


if __name__ == "__main__":
    main()
