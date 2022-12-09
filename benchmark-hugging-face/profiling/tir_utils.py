
def get_ir_mod(name):
  if name == "D1":
    from default_tir_1 import ModuleD1
    return ModuleD1
  elif name == "D2":
    from default_tir_2 import ModuleD2
    return ModuleD2
  elif name == "BaseAdd":
    from base_add_tir import BaseAdd
    return BaseAdd
  else:
    raise NotImplementedError("Other kernels except D1, D2 are not supported")

def get_rnd_inputs(name):
  if name == "D1":
    from default_tir_1 import get_rnd_inputs
    return get_rnd_inputs()
  elif name == "D2":
    from default_tir_2 import get_rnd_inputs
    return get_rnd_inputs()
  elif name == "BaseAdd":
    from base_add_tir import get_rnd_inputs
    return get_rnd_inputs()
  else:
    raise NotImplementedError("Random inputs to kernels except D1, D2 are not supported")
