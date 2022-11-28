import os
import tempfile
import copy
from functools import partial

import tvm
from tvm import autotvm
from tvm import relay, auto_scheduler
from tvm import meta_schedule as ms
from tvm.relay import vm
from tvm.relay.backend import Executor
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
META_TYPE = "Meta"
def get_tvm_vm(mod,
               opt_level,
               target,
               target_host,
               params,
               dev,
               nhwc=False,
               tuning_logfile="",
               tuning_type=ANSOR_TYPE,
              ):
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

def tvm_test(
      benchmark_test,
      onnx_model,
      inputs,
      opt_level,
      target,
      target_host,
      freeze=True,
      tuning_logs=""):
  print("----- TVM testing -----")
  shape_dict = {input_name: input.shape for (input_name, input) in inputs.items()}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=freeze)
  mod = relay.transform.DynamicToStatic()(mod)

  tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in inputs.items()}

  dev = tvm.device(str(target), 0)

  m = get_tvm_vm(
        mod,
        opt_level,
        target,
        target_host,
        params,
        dev,
        tuning_logfile=tuning_logs,
      )
  m.set_input("main", **tvm_inputs)
  tvm_runner = partial(m.invoke, "main")
  benchmark_test(tvm_runner, framework_name = "TVM")

def tvm_ansor_tuning(mod, target, target_host, params, trials_num, log_dir, model_name):
  log_file = str(log_dir.joinpath(model_name).with_suffix("_tuned.json"))
  # extract workloads from relay program
  print("Extract tasks...")
  tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target=target, target_host=target_host)
  for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

  print("Begin auto-scheduler tuning...")
  tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
  tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=trials_num,  # change this to 20000 to achieve the best performance
    runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
  )

  tuner.tune(tune_option)
  print("Tuning finished")

def tvm_meta_tuning(mod, params, target, trials_num, log_dir, model_name):
  module_equality="ignore-ndarray"
  workload_path = str(log_dir.joinpath( model_name + "_workload.json"))
  record_path = str(log_dir.joinpath( model_name + "_records.json"))
  # TODO(vvchernov): is "graph" tag related to graph executor?
  executor = Executor("graph", {"link-params": True}) # "aot"
  # This line is necessary for link-params to take effect during
  # task extraction and relay.build(...).
  mod = mod.with_attr("executor", executor)
  # Empty data base with workload and record file names
  database = ms.database.JSONDatabase(
    workload_path,
    record_path,
#    work_dir=log_dir,
    module_equality=module_equality,
  )
  print("Begin meta-scheduler tuning...")
  with tempfile.TemporaryDirectory() as work_dir:
    database = ms.relay_integration.tune_relay(
      mod=mod,
      target=target,
      params=params,
      work_dir=work_dir,
      # for faster tuning
      max_trials_global=trials_num,
      max_trials_per_task=8,
      num_trials_per_iter=8,
      strategy="replay-trace",  # TODO(vvchernov): "evolutionary",
      database=database,
      # Without this, the same workloads with different constant weights
      # are treated as distinct tuning tasks.
      module_equality=module_equality,
    )

  # vm_lib = ms.relay_integration.compile_relay(
  #   database=database,
  #   mod=mod,
  #   target=target,
  #   params=params,
  #   backend="vm", # "graph" by default
  # )
  print("Tuning finished")
  pass
