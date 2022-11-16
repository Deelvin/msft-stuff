import os
import copy

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import vm
from tvm import autotvm
from tvm.runtime import vm as tvm_rt_vm


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

def tvm_test(benchmark_test, onnx_model, inputs, opt_level, target, target_host, freeze=True):
  print("----- TVM testing -----")
  shape_dict = {input_name: input.shape for (input_name, input) in inputs}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=freeze)
  mod = relay.transform.DynamicToStatic()(mod)

  tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in inputs}

  dev = tvm.device(str(target), 0)

  m = get_tvm_vm(mod, opt_level, target, target_host, params, dev)
  m.set_input(**tvm_inputs)
  benchmark_test(m.run, framework_name = "TVM")
