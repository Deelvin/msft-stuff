import argparse
from pathlib import Path

import onnx

import tvm

from utils.tvm_utils import get_distilbert_mod_params, tvm_ansor_tuning, tvm_meta_tuning
from utils.utils import SKYLAKE_TARGET


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
  parser.add_argument("-n", "--trials_number", default=20000, type=int, help=\
    "Maximal number of trials for model tuning")
  parser.add_argument("-npt", "--trials_per_task_number", default=1000, type=int, help=\
    "Number of trials per task for model tuning")
  parser.add_argument("-t", "--target", default=SKYLAKE_TARGET, type=str, help=\
    "Target for model inference")
  parser.add_argument("-o", "--opt_level", default=3, type=int, help=\
    "Optimization level for TVM compilation")
  parser.add_argument("-a", "--artificial_input", action="store_true", default=False, help=\
    "Artificially generated inputs. if false the default text from utils is tokenized")
  parser.add_argument("--meta", action="store_true", default=False, help=\
    "Switch on meta scheduler, by default it is auto scheduler")
  parser.add_argument("-d", "--use_dnnl", action="store_true", default=False, help=\
    "Switch on using DNNL")
  parser.add_argument("-e", "--extracted_task_indices", nargs="*", type=int, default=[], help=\
    "Indices of task which should be extracted form the model for tuning. Need for effective debugging")
  parser.add_argument("-ex", "--excluded_task_indices", nargs="*", type=int, default=[], help=\
    "Indices of task which should be excluded form the model for tuning. Need for effective debugging")

  args = parser.parse_args()

  onnx_model = onnx.load(args.model_path)

  mod, params = get_distilbert_mod_params(
                  onnx_model,
                  args.artificial_input,
                  args.opt_level,
                  use_dnnl=args.use_dnnl,
                )
  model_path = Path(args.model_path)
  model_name = model_path.stem

  print("----- TVM tuning -----")
  if args.meta:
    # Model tuning by tvm meta-scheduler
    log_dir = model_path.parent.joinpath(model_name + "_meta")
    log_dir.mkdir(parents=True, exist_ok=True)
    tvm_meta_tuning(
      mod,
      params,
      tvm.target.Target(args.target, args.target),
      args.opt_level,
      trials_num=args.trials_number,
      trials_per_task_num=args.trials_per_task_number,
      log_dir=log_dir,
      task_indices=args.extracted_task_indices,
      exl_task_indices=args.excluded_task_indices,
    )
  else:
    # Model tuning by tvm auto-scheduler
    log_dir = model_path.parent.joinpath(model_name + "_ansor")
    log_dir.mkdir(parents=True, exist_ok=True)
    tvm_ansor_tuning(
      mod,
      args.target,
      args.target,
      params,
      trials_num=args.trials_number,
      log_dir=log_dir,
      model_name=model_name,
    )
  print("----- TVM tuning finished -----")
