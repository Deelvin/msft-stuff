import typing

import numpy

import utils.benchmark_base
import utils.inference
import utils.load


class OnnxModel(utils.benchmark_base.ModelBase):
    def __init__(self, model_path: str, input_dict: typing.Dict[str, numpy.ndarray]):
        self._engine = utils.load.load_onnxruntime(model_path)
        self._input_dict = input_dict

    def prepare_input(self) -> typing.Dict[str, numpy.ndarray]:
        return self._input_dict

    def get_inference_function(self, inputs: typing.Dict[str, numpy.ndarray]):
        runner = utils.inference.ort_inference_runner(self._engine)
        return lambda: runner(inputs)
