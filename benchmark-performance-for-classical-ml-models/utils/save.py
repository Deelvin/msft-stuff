import os
import pickle
import typing

import decorator
import onnx
import treelite


def model_saver(
    save_root: str, save_function: typing.Callable[[typing.Any, str], None]
):
    @decorator.decorator
    def wrapper(
        model_factory: typing.Callable[..., typing.Tuple[typing.Any, str]],
        *args,
        **kwargs
    ) -> str:
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        model, save_name = model_factory(*args, **kwargs)
        save_path = os.path.join(save_root, save_name)
        save_function(model, save_path)

        return save_path

    return wrapper


def pickle_saver(save_root: str):
    def save_to_file(model: typing.Any, save_path: str) -> None:
        with open(save_path, "wb") as f:
            f.write(pickle.dumps(model))

    return model_saver(save_root, save_to_file)


def sklearn_saver(save_root: str):
    return pickle_saver(save_root)


def xgboost_saver(save_root: str):
    def save_to_file(model: typing.Any, save_path: str) -> None:
        model.save_model(save_path)

    return model_saver(save_root, save_to_file)


def onnx_saver(save_root: str):
    def save_to_file(model: onnx.ModelProto, save_path: str) -> None:
        with open(save_path, "wb") as f:
            f.write(model.SerializeToString())

    return model_saver(save_root, save_to_file)


def treelite_saver(save_root: str):
    def save_to_file(model: treelite.frontend.Model, save_path: str) -> None:
        model.export_lib(
            libpath=save_path,
            toolchain="gcc",
            verbose=True,
            params=dict(parallel_comp=4),
        )

    return model_saver(save_root, save_to_file)
