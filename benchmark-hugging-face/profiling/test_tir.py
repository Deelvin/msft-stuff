from pathlib import Path
import argparse
from functools import partial

import tvm
from tvm import meta_schedule as ms

from utils.utils import SKYLAKE_TARGET, perf_test
from tir_utils import get_ir_mod, get_rnd_inputs


def main():
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Benchmark separated kernels implemented by TIR. " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-k", "--kernel_name", default="Dense", type=str, help=\
    "The name of kernel implemented by TIR")
  parser.add_argument("-t", "--target", default=SKYLAKE_TARGET, type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")
  parser.add_argument("-m", "--use_meta", action="store_true", default=False, help=\
    "Switch on using of statistics from meta-scheduler by kernel during compilation")
  parser.add_argument("-md", "--meta_dir", default="./kernel_logs", type=str, help=\
    "The path to directory with statistics for kernel implemented by TIR. " +
    "Statistics are located in directory with kernel name inside the mets directory")
  parser.add_argument("-s", "--save_dir", default="./kernel_libs", type=str, help=\
    "The path to save directory. If it is not empty the shared library of kernel (<kernel_name>_lib.so) is tried to save there")

  args = parser.parse_args()

  target = tvm.target.Target(args.target, host=args.target)
  name = args.kernel_name

  ir_mod = get_ir_mod(name)
  if args.use_meta:
    # get kernel after transformations by meta-scheduler
    from utils.meta_utils import get_json_database
    meta_dir = Path(args.meta_dir).joinpath(name)
    meta_dir.mkdir(parents=True, exist_ok=True)

    database = get_json_database(meta_dir)

    shed = ms.tir_integration.compile_tir(
            database=database,
            mod=ir_mod,
            target=target,
           )

    rt_lib = tvm.build(shed.mod, target=target)
  else:
    rt_lib = tvm.build(ir_mod, target=target)

  # Save shared library from kernel if need
  if args.save_dir:
    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    so_path = save_dir_path.joinpath(name + "_lib.so")
    rt_lib.export_library(so_path)

  func = rt_lib["main"]

  inputs = get_rnd_inputs(name)
  kernel_runner = partial(func, *inputs)
  perf_test(kernel_runner, args.iters_number, name)


if __name__ == "__main__":
  main()
