import argparse
from pathlib import Path

import onnx

import tvm
from tvm import relay

from utils import get_distilbert_inputs
from tvm_utils import tvm_ansor_tuning, tvm_meta_tuning


if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Tuning of distilbert-like model by auto-scheduler from TVM",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to onnx-file with model")
  parser.add_argument("-n", "--trials_number", default=100, type=int, help=\
    "Number of iterations of inference for performance measurement")
  parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
    "Target for model inference")
  parser.add_argument("-a", "--artificial_input", action="store_true", default=False, help=\
    "Artificially generated inputs. if false the default text from utils is tokenized")
  parser.add_argument("--meta", action="store_true", default=False, help=\
    "Switch on meta scheduler, by default it is auto scheduler")
  parser.add_argument("-e", "--extracted_task_indices", nargs="*", type=int, default=[], help=\
    "Indices of task which should be extracted form the model for tuning. Need for effective debugging")

  args = parser.parse_args()

  onnx_model = onnx.load(args.model_path)

  encoded_inputs = get_distilbert_inputs(args.artificial_input)

  print("----- TVM tuning -----")
  shape_dict = {input_name: input.shape for (input_name, input) in encoded_inputs.items()}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
  mod = relay.transform.DynamicToStatic()(mod)

  model_path = Path(args.model_path)
  model_name = model_path.stem
  log_dir = model_path.parent

  if args.meta:
    # Model tuning by tvm meta-scheduler
    tvm_meta_tuning(
      mod,
      params,
      tvm.target.Target(args.target),
      trials_num=args.trials_number,
      log_dir=log_dir,
      model_name=model_name,
      task_indices=args.extracted_task_indices,
    )
  else:
    # Model tuning by tvm auto-scheduler
    tvm_ansor_tuning(
      mod,
      args.target,
      args.target,
      params,
      trials_num=args.trials_number,
      log_dir=log_dir,
      model_name=model_name,
    )
