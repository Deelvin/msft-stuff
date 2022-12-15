import argparse
from pathlib import Path
from functools import partial

import onnx

from utils.utils import SKYLAKE_TARGET, perf_test, get_distilbert_inputs


if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Performance test of Distilbert with text on TVM using VirtualMachine or ONNX Runtime (CPU EP). " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to Distilbert-base-cased model in onnx format")
  parser.add_argument("-i", "--input_text", default="", type=str, help=\
    "Test input text for Distilbert-base-cased model")
  parser.add_argument("-tvm", action="store_false", default=True, help=\
    "Performance test of TVM")
  parser.add_argument("-ort", action="store_true", default=False, help=\
    "Performance test of ONNX Runtime")
  parser.add_argument("-t", "--target", default=SKYLAKE_TARGET, type=str, help=\
    "Target for model inference")
  parser.add_argument("-o", "--opt_level", default=3, type=int, help=\
    "Optimization level for TVM compilation")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")
  parser.add_argument("-tu", "--tuning_logs", default="", type=str, help=\
    "The path to tuning logs. It can be json file for auto-scheduler or directory for meta-scheduler")
  parser.add_argument("-meta", action="store_true", default=False, help=\
    "Use meta-scheduler database files for compilation")
  parser.add_argument("-a", "--artificial_input", action="store_true", default=False, help=\
    "Artificially generated inputs. if false the default text from utils is tokenized")

  args = parser.parse_args()

  freeze = True

  onnx_model = onnx.load(args.model_path)

  encoded_inputs = {}
  if args.input_text == "":
    encoded_inputs = get_distilbert_inputs(args.artificial_input)
  else:
    encoded_inputs = get_distilbert_inputs(args.artificial_input, args.input_text)

  benchmark_test = partial(perf_test, iters_number = args.iters_number, model_name = "Distilbert-text")

  if args.tvm:
    from utils.tvm_utils import tvm_test
    tvm_test(
      benchmark_test,
      onnx_model,
      encoded_inputs,
      args.opt_level,
      args.target,
      args.target,
      freeze,
      args.tuning_logs,
      args.meta,
      model_name=Path(args.model_path).stem
    )

  if args.ort:
    from utils.ort_utils import ort_test
    ort_test(benchmark_test, args.model_path, encoded_inputs)
