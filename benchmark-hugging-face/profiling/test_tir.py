from pathlib import Path
import argparse
from functools import partial

import tvm
from tvm import meta_schedule as ms

from default_tir_1 import ModuleD1

from utils.utils import perf_test


def get_ir_mod(name):
  if name == "D1":
    return ModuleD1["main"]
  else:
    raise NotImplementedError("Other kernels except D1 are not supported")

def main():
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Benchmark separated kernels implemented by TIR. " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-k", "--kernel_name", default="D1", type=str, help=\
    "The name of kernel implemented by TIR")
  parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")
  parser.add_argument("-m", "--meta_dir", default=None, type=str, help=\
    "The path to directory with statistics for kernel implemented by TIR")
  parser.add_argument("-s", "--save_dir", default="./kernel_libs", type=str, help=\
    "The path to save directory. If it is not empty the shared library of kernel (<kernel_name>_lib.so) is tried to save there")

  args = parser.parse_args()

  target = tvm.target.Target(args.target, host=args.target)
  name = args.kernel_name

  ir_mod = get_ir_mod(name)
  if args.meta_dir:
    # get kernel after transformations by meta-scheduler
    from utils.meta_utils import MODULE_EQUALITY, get_workload_path, get_record_path, get_work_dir
    workload_path = get_workload_path(args.meta_dir)
    records_path = get_record_path(args.meta_dir)
    work_dir = get_work_dir(args.meta_dir)
    database = ms.database.JSONDatabase(
      path_workload=workload_path,
      path_tuning_record=records_path,
      work_dir=work_dir,
      module_equality=MODULE_EQUALITY,
    )
    
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
    save_dir_path = Path(args.save_dir).joinpath(name)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    so_path = save_dir_path.joinpath(name + "_lib.so")
    rt_lib.save(str(so_path))

  func = rt_lib["main"]
  
  inputs = ir_mod.get_rnd_inputs()
  kernel_runner = partial(func, *inputs)
  perf_test(kernel_runner, args.iters_number, name)


if __name__ == "__main__":
  main()
