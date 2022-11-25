import os
import pickle
import time
import typing

import decorator
import onnx
import treelite

sklearn_classifiers = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
]


sklearn_regressors = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
]

outlier_detectors = [
    "IsolationForest",
]


class Profiler:
    def __init__(self, name="", show_latency=False, iterations_number=1):
        self.name = name
        self.show_latency = show_latency
        self.iterations_number = iterations_number

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type, value, traceback):
        end_time = time.perf_counter()
        if self.show_latency:
            print(
                "[{}] elapsed time: {:.3f} ms ({} iterations)".format(
                    self.name,
                    (end_time - self.start_time) * 1000 / self.iterations_number,
                    self.iterations_number,
                )
            )


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


def sklearn_saver(save_root: str):
    def save_to_file(model: typing.Any, save_path: str) -> None:
        with open(save_path, "wb") as f:
            f.write(pickle.dumps(model))

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
