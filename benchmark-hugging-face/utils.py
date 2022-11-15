import os
import time
import copy

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import vm
from tvm import autotvm
from tvm.runtime import vm as tvm_rt_vm


def perf_test(run, iters_number = 1000, model_name = "ResNet50-v1", framework_name = "TVM+VM"):
    assert iters_number > 0

    tic = time.perf_counter()
    for i in range(iters_number):
        run()
    toc = time.perf_counter()
    print(f"Elapsed time: {toc - tic:0.4f} seconds for {iters_number} iterations of inference of {model_name} model by {framework_name}")

def get_vm_lib(irmod, target, target_host, params):
    return vm.compile(
              copy.deepcopy(irmod),
              target,
              params=params,
              target_host=target_host,
           )

ANSOR_TYPE = "Ansor"
AUTO_TVM_TYPE = "AutoTVM"
def get_tvm_vm(mod,
               opt_level,
               target,
               target_host,
               params,
               dev,
               nhwc = False,
               tuning_logfile = "",
               tuning_type = ANSOR_TYPE):
  if tuning_logfile == "":
    tuning_logfile = os.getenv("AUTOTVM_TUNING_LOG")
  lib = None
  if tuning_logfile:
    print("Use tuning file from ", tuning_logfile, ": ", tuning_logfile)
    if tuning_type == ANSOR_TYPE:
      desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv2d_transpose": ["NHWC", "default"],
        "nn.upsampling": ["NHWC", "default"],
        "vision.roi_align": ["NHWC", "default"],
      }
      with auto_scheduler.ApplyHistoryBest(tuning_logfile):
        with tvm.transform.PassContext(
          opt_level=opt_level,
          config={
            "relay.backend.use_auto_scheduler": True,
            "relay.FuseOps.max_depth": 30,
          }
          ):
          if nhwc:
            mod = relay.transform.InferType()(mod)
            model_nhwc = relay.transform.ConvertLayout(desired_layouts)(mod)
            model_nhwc = relay.transform.EliminateCommonSubexpr()(model_nhwc)
            mod = relay.transform.FoldConstant()(model_nhwc)
          lib = get_vm_lib(mod, target, target_host, params)
    elif tuning_type == AUTO_TVM_TYPE:
      with relay.build_config(opt_level=opt_level):
        with autotvm.apply_history_best(tuning_logfile):
          lib = get_vm_lib(mod, target, target_host, params)
    else:
      # TODO(vvchernov): replace prints by logger, but investigate ORT logging system for python before
      # print is not commented out while it declares error
      print("ERROR: Tuning log type {} is unsupported. ".format(tuning_type),
            "Only {} and {} types are supported".format(ANSOR_TYPE, AUTO_TVM_TYPE))
      return None
  else:
    with tvm.transform.PassContext(opt_level=opt_level):
      lib = get_vm_lib(mod, target, target_host, params)

  return tvm_rt_vm.VirtualMachine(lib, dev)

def get_onnx_input_name(model):
  inputs = [node.name for node in model.graph.input]
  initializer = [node.name for node in model.graph.initializer]

  inputs = list(set(inputs) - set(initializer))
  return inputs
