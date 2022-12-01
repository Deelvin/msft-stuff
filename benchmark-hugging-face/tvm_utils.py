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
from meta_utils import MODULE_EQUALITY, get_workload_path, get_record_path


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
               tuning_log="",
               tuning_type=ANSOR_TYPE,
               model_name="",
              ):
  if tuning_log == "":
    tuning_log = os.getenv("AUTOTVM_TUNING_LOG")
  lib = None
  if tuning_log:
    if tuning_type == ANSOR_TYPE:
      print("Use tuning file from ", tuning_log)
      desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv2d_transpose": ["NHWC", "default"],
        "nn.upsampling": ["NHWC", "default"],
        "vision.roi_align": ["NHWC", "default"],
      }
      with auto_scheduler.ApplyHistoryBest(tuning_log):
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
      print("Use tuning file from ", tuning_log)
      with relay.build_config(opt_level=opt_level):
        with autotvm.apply_history_best(tuning_log):
          lib = get_vm_lib(mod, target, target_host, params)
    elif tuning_type == META_TYPE:
      print("Use tuning files from directory:", tuning_log)
      workload_path = get_workload_path(tuning_log, model_name)
      record_path = get_record_path(tuning_log, model_name)
      # Import data base from workload and record files
      database = ms.database.JSONDatabase(
        workload_path,
        record_path,
        module_equality=MODULE_EQUALITY,
      )

      lib = ms.relay_integration.compile_relay(
        database=database,
        mod=mod,
        target=target,
        params=params,
        backend="vm", # "graph" by default
        opt_level=opt_level,
      )
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
      tuning_logs="",
      use_meta=False,
      model_name="",
    ):
  print("----- TVM testing of", model_name, "-----")
  shape_dict = {input_name: input.shape for (input_name, input) in inputs.items()}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=freeze)
  mod = relay.transform.DynamicToStatic()(mod)

  tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in inputs.items()}

  dev = tvm.device(str(target), 0)
  if use_meta:
    tuning_type = META_TYPE
  else:
    tuning_type = ANSOR_TYPE
  m = get_tvm_vm(
        mod,
        opt_level,
        target,
        target_host,
        params,
        dev,
        tuning_log=tuning_logs,
        tuning_type=tuning_type,
        model_name=model_name,
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

def tune_relay_with_task_extractor(
      mod,
      target,
      params,
      database,
      module_equality,
      trials_num = 20000,
      max_trials_per_task=8,
      num_trials_per_iter=8,
      strategy="replay-trace",  # TODO(vvchernov): "evolutionary",
      extracted_task_indices=[],
      excluded_task_indices=[],
    ):
  space = "post-order-apply"
  seed = None
  builder = "local"
  runner = "local"
  cost_model = "xgb"
  measure_callbacks = "default"
  task_scheduler = "gradient"
  with tempfile.TemporaryDirectory() as work_dir:
    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
      extracted_tasks=ms.relay_integration.extract_tasks(
                        mod,
                        target,
                        params,
                        module_equality=module_equality),
      work_dir=work_dir,
      space=space,
      strategy=strategy,
      seed=seed,
    )

    if extracted_task_indices and excluded_task_indices:
      raise BrokenPipeError("Both lists of indices (extracted or excluded tasks) can not exist simultaneously!")

    filtered_tasks=[]
    filtered_task_weights=[]
    if extracted_task_indices:
      filtered_tasks = [tasks[i] for i in extracted_task_indices]
      filtered_task_weights = [task_weights[i] for i in extracted_task_indices]
    else:
      filtered_tasks = tasks
      filtered_task_weights = task_weights

    if  excluded_task_indices:
      filtered_tasks = [tasks[i] for i in range(len(tasks)) if i not in excluded_task_indices]
      filtered_task_weights = [task_weights[i] for i in range(len(task_weights)) if i not in excluded_task_indices]
    else:
      filtered_tasks = tasks
      filtered_task_weights = task_weights

    ms.relay_integration.tune_tasks(
        tasks=filtered_tasks,
        task_weights=filtered_task_weights,
        work_dir=work_dir,
        max_trials_global=trials_num,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        module_equality=module_equality,
    )

def tvm_meta_tuning(
      mod,
      params,
      target,
      trials_num,
      log_dir,
      model_name,
      task_indices=[],
      exl_task_indices=[],
    ):
  # Without this, the same workloads with different constant weights
  # are treated as distinct tuning tasks.
  workload_path = get_workload_path(log_dir, model_name)
  record_path = get_record_path(log_dir, model_name)
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
    module_equality=MODULE_EQUALITY,
  )
  print("Begin meta-scheduler tuning...")
  tune_relay_with_task_extractor(
    mod=mod,
    target=target,
    params=params,
    database=database,
    module_equality=MODULE_EQUALITY,
    trials_num=trials_num,
    extracted_task_indices=task_indices,
    excluded_task_indices=exl_task_indices,
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
