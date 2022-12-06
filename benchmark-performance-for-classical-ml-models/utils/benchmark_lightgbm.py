import typing

import numpy

import utils.benchmark_base
import utils.inference
import utils.load


class LightgbmModel(utils.benchmark_base.ModelBase):
    def __init__(self, model_path: str, input_dict: typing.Dict[str, numpy.ndarray]):
        print(f"LightgbmModel: {model_path}")
        self._engine = utils.load.load_lightgbm(model_path)
        self._input_dict = input_dict

    def prepare_input(self) -> numpy.ndarray:
        return list(self._input_dict.values())[0]

    def get_inference_function(self, inputs: typing.Dict[str, numpy.ndarray]):
        runner = utils.inference.lightgbm_inference_runner(self._engine)
        return lambda: runner(inputs)
