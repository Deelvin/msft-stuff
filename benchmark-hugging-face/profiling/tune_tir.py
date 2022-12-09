from pathlib import Path
import argparse

import tvm
from tvm import meta_schedule as ms

from utils.meta_utils import MODULE_EQUALITY, get_workload_path, get_record_path, get_work_dir
from utils.utils import SKYLAKE_TARGET
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
  parser.add_argument("-t", "--target", default=SKYLAKE_TARGET, type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--trials_number", default=2048, type=int, help=\
    "Maximal number of trials for model tuning")
  parser.add_argument("-l", "--log_dir", default="./kernel_logs", type=str, help=\
    "The path to directory with tuning statistics for the kernel")

  args = parser.parse_args()

  target = tvm.target.Target(args.target, args.target)
  name = args.kernel_name
  log_dir = Path(args.log_dir).joinpath(name)
  log_dir.mkdir(parents=True, exist_ok=True)

  workload_path = get_workload_path(log_dir)
  record_path = get_record_path(log_dir)
  work_dir = get_work_dir(log_dir)
  # Empty data base with workload and record file names
  database = ms.database.JSONDatabase(
    workload_path,
    record_path,
    work_dir=work_dir,
    module_equality=MODULE_EQUALITY,
  )

  ir_mod = get_ir_mod(name)

  print("----- Kernel TIR tuning -----")
  ms.tir_integration.tune_tir(
      mod=ir_mod,
      target=target,
      work_dir=str(work_dir),
      max_trials_global=args.trials_number,
      database=database,
      strategy="replay-trace",
  )
  print("----- Kernel TIR tuning finished -----")


if __name__ == "__main__":
  main()
