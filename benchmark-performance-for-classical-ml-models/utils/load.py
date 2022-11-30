import pickle
import typing

import onnxruntime
import treelite_runtime


def load_pickle(model_path: str) -> typing.Any:
    with open(model_path, "rb") as model_file:
        model = pickle.loads(model_file.read())

    return model


def load_sklearn(model_path: str) -> typing.Any:
    return load_pickle(model_path)


def load_xgboost(model_path: str) -> typing.Any:
    return load_pickle(model_path)


def load_lightgbm(model_path: str) -> typing.Any:
    return load_pickle(model_path)


def load_onnxruntime(model_path: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(model_path)


def load_treelite(model_path: str) -> treelite_runtime.Predictor:
    return treelite_runtime.Predictor(libpath=model_path)
