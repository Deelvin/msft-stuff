
def get_ir_mod(name):
  if name == "D1":
    from default_tir_1 import ModuleD1
    return ModuleD1
  else:
    raise NotImplementedError("Other kernels except D1 are not supported")

def get_rnd_inputs(name):
  if name == "D1":
    from default_tir_1 import get_rnd_inputs
    return get_rnd_inputs()
  else:
    raise NotImplementedError("Random inputs to kernels except D1 are not supported")
