import os
import pickle
import time
import typing

import onnx

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


class ModelSaver:
    def __init__(self, save_root: str):
        self.save_root = save_root

    def __call__(
        self,
        convert_func: typing.Callable[..., typing.Tuple[typing.Any, str]],
    ):
        def wrapper(*args, **kwargs) -> str:
            if not os.path.exists(self.save_root):
                os.makedirs(self.save_root)

            model, save_name = convert_func(*args, **kwargs)

            save_path = os.path.join(self.save_root, save_name)
            with open(save_path, "wb") as f:
                f.write(self._serialize_model(model))

            return save_path

        return wrapper

    def _serialize_model(self, model: typing.Any) -> bytes:
        raise NotImplementedError("ModelSaver is abstract class")


class SaveSKLearn(ModelSaver):
    def __init__(self, save_root: str):
        super().__init__(save_root)

    def _serialize_model(self, model: typing.Any) -> bytes:
        return pickle.dumps(model)


class SaveONNX(ModelSaver):
    def __init__(self, save_root: str):
        super().__init__(save_root)

    def _serialize_model(self, model: onnx.ModelProto) -> bytes:
        return model.SerializeToString()
