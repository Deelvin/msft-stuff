
from default_tir_1 import ModuleD1


def get_ir_mod(name):
  if name == "D1":
    return ModuleD1["main"]
  else:
    raise NotImplementedError("Other kernels except D1 are not supported")
