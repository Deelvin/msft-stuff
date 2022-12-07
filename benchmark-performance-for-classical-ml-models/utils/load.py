import pickle
import typing

import onnxruntime
import treelite_runtime

import utils.common


def load_pickle(model_path: str) -> typing.Any:
    with open(model_path, "rb") as model_file:
        model = pickle.loads(model_file.read())

    return model


def load_sklearn(model_path: str) -> typing.Any:
    model = load_pickle(model_path)
    model.set_params(n_jobs=utils.common.NUM_THREADS)
    return model


def load_xgboost(model_path: str) -> typing.Any:
    model = load_pickle(model_path)
    model.set_params(n_jobs=utils.common.NUM_THREADS)
    return model


def load_lightgbm(model_path: str) -> typing.Any:
    model = load_pickle(model_path)
    model.set_params(n_jobs=utils.common.NUM_THREADS)
    return model


def load_onnxruntime(model_path: str) -> onnxruntime.InferenceSession:
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = utils.common.NUM_THREADS
    return onnxruntime.InferenceSession(model_path, sess_options=sess_options)


def load_treelite(model_path: str) -> treelite_runtime.Predictor:
    return treelite_runtime.Predictor(
        libpath=model_path, nthread=utils.common.NUM_THREADS
    )
