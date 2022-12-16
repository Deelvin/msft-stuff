import argparse
from pathlib import Path

import onnx

from utils.utils import ART_SIZE, SKYLAKE_TARGET, get_distilbert_inputs
from utils.tvm_utils import tvm_profile


if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Profiling test of Distilbert with text on TVM using VirtualMachineProfiler. " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to Distilbert-base-cased model in onnx format")
  parser.add_argument("-i", "--input_text", default="", type=str, help=\
    "Test input text for Distilbert-base-cased model")
  parser.add_argument("-t", "--target", default=SKYLAKE_TARGET, type=str, help=\
    "Target for model inference")
  parser.add_argument("-o", "--opt_level", default=3, type=int, help=\
    "Optimization level for TVM compilation")
  parser.add_argument("-tu", "--tuning_logs", default="", type=str, help=\
    "The path to tuning logs. It can be json file for auto-scheduler or directory for meta-scheduler")
  parser.add_argument("-meta", action="store_true", default=False, help=\
    "Use meta-scheduler database files for compilation")
  parser.add_argument("-d", "--use_dnnl", action="store_true", default=False, help=\
    "Switch on using DNNL")
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

  prefix = "_" + str(ART_SIZE) if args.artificial_input else ""
  tvm_profile(
    onnx_model,
    encoded_inputs,
    args.target,
    args.target,
    opt_level=args.opt_level,
    freeze=freeze,
    tuning_logs=args.tuning_logs,
    use_meta=args.meta,
    use_dnnl=args.use_dnnl,
    model_name=Path(args.model_path).stem + prefix
  )
