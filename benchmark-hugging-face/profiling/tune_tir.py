from pathlib import Path
import argparse

import tvm

from utils.tvm_utils import tvm_meta_tuning
from tir_utils import get_ir_mod


def main():
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Tuning by meta-scheduler of separated kernels implemented by TIR. " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-k", "--kernel_name", default="D1", type=str, help=\
    "The name of kernel implemented by TIR")
  parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--trials_number", default=20000, type=int, help=\
    "Maximal number of trials for model tuning")
  parser.add_argument("-npt", "--trials_per_task_number", default=1000, type=int, help=\
    "Number of trials per task for model tuning")
  parser.add_argument("-l", "--log_dir", default="./kernel_logs", type=str, help=\
    "The path to directory with tuning statistics for the kernel")

  args = parser.parse_args()

  target = tvm.target.Target(args.target, args.target)
  name = args.kernel_name

  ir_mod = get_ir_mod(name)
  
  print("----- Kernel tuning -----")
  # Model tuning by tvm meta-scheduler
  log_dir = Path(args.log_dir).joinpath(name)
  log_dir.mkdir(parents=True, exist_ok=True)
  tvm_meta_tuning(
    ir_mod,
    None,
    target,
    trials_num=args.trials_number,
    trials_per_task_num=args.trials_per_task_number,
    log_dir=log_dir,
  )
  print("----- Kernel tuning finished -----")


if __name__ == "__main__":
  main()
