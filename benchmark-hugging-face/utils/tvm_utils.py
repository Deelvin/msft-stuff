import os
import copy
from functools import partial

import tvm
from tvm import autotvm
from tvm import relay, auto_scheduler
from tvm import meta_schedule as ms
from tvm.relay import vm
from tvm.relay.op.contrib import dnnl
from tvm.relay.backend import Executor
from tvm.runtime import vm as tvm_rt_vm
from tvm.runtime import profiler_vm
# Preregistration
from tvm.tir.tensor_intrin import *

from utils.utils import DISTILBERT_TEST_TEXT, get_distilbert_inputs
from utils.meta_utils import (
  MODULE_EQUALITY,
  TUNE_SPACE,
  TUNE_STRATEGY,
  TUNE_SEED,
  TUNE_BUILDER,
  TUNE_RUNNER,
  TUNE_COST_MODEL,
  TUNE_MEASURE_CALLBACKS,
  TUNE_TASK_SCHEDULER,
  TUNE_NUM_TRIALS_PER_ITER,
  get_work_dir,
  get_json_database,
)


def get_distilbert_mod_params_with_inputs(onnx_model,
                                          inputs,
                                          opt_level,
                                          freeze=True,
                                          dnnl_enabled=False,
                                          prune_subgraphs=True,):
  shape_dict = {input_name: input.shape for (input_name, input) in inputs.items()}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=freeze)
  mod = relay.transform.DynamicToStatic()(mod)

  # See details in https://github.com/apache/tvm/blob/e9b331831aef4f4866c719b8bd66b436038b3496/tests/python/contrib/test_dnnl.py#L61
  # conv support was skipped
  if dnnl_enabled:
    mod = dnnl.rewrite_layer_norm(mod)
    mod = dnnl.rewrite_dense_bias_gelu_reshape_last(mod)
    mod = dnnl.legalize_qnn_for_dnnl(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            relay.transform.MergeComposite(dnnl.pattern_table()),
            relay.transform.AnnotateTarget("dnnl"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=opt_level):
        mod = byoc_seq(mod)
        if prune_subgraphs:
            mod = dnnl.prune_dnnl_subgraphs(mod)

  # Set attr which insert const tensor in TIR
  executor = Executor("graph", {"link-params": True}) # "aot"
  mod = mod.with_attr("executor", executor)

  return mod, params

def get_distilbert_mod_params(onnx_model,
                              artificial: bool,
                              opt_level: int,
                              input_text : str = DISTILBERT_TEST_TEXT,
                              tag : str = "distilbert-base-uncased",
                              use_dnnl: bool = False,
                              prune: bool = True):
  encoded_inputs = get_distilbert_inputs(artificial, input_text, tag)

  return get_distilbert_mod_params_with_inputs(
            onnx_model,
            encoded_inputs,
            opt_level,
            freeze=True,
            dnnl_enabled=use_dnnl,
            prune_subgraphs=prune,
         )

def get_vm_lib(irmod, target, params):
    return vm.compile(
              copy.deepcopy(irmod),
              target,
              params=params,
           )

ANSOR_TYPE = "Ansor"
AUTO_TVM_TYPE = "AutoTVM"
META_TYPE = "Meta"
def get_tvm_tuned_vm_lib(
      mod,
      opt_level,
      target,
      params,
      nhwc=False,
      tuning_log="",
      tuning_type=ANSOR_TYPE,
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
          lib = get_vm_lib(mod, target, params)
    elif tuning_type == AUTO_TVM_TYPE:
      print("Use tuning file from ", tuning_log)
      with relay.build_config(opt_level=opt_level):
        with autotvm.apply_history_best(tuning_log):
          lib = get_vm_lib(mod, target, params)
    elif tuning_type == META_TYPE:
      print("Use tuning files from directory:", tuning_log)
      database = get_json_database(tuning_log)

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
      lib = get_vm_lib(mod, target, params)

  return lib

def get_tvm_vm(mod,
               opt_level,
               target,
               params,
               dev,
               nhwc=False,
               tuning_log="",
               tuning_type=ANSOR_TYPE,
              ):
  lib = get_tvm_tuned_vm_lib(
          mod,
          opt_level,
          target,
          params,
          nhwc=nhwc,
          tuning_log=tuning_log,
          tuning_type=tuning_type,
        )
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
      use_dnnl=False,
      prune=True,
      model_name="",
    ):
  print("----- TVM testing of", model_name, "-----")
  mod, params = get_distilbert_mod_params_with_inputs(
                  onnx_model,
                  inputs,
                  opt_level,
                  freeze=freeze,
                  dnnl_enabled=use_dnnl,
                  prune_subgraphs=prune,
                )

  tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in inputs.items()}

  dev = tvm.device(str(target), 0)
  tvm_target = tvm.target.Target(target, target_host)
  if use_meta:
    tuning_type = META_TYPE
  else:
    tuning_type = ANSOR_TYPE
  m = get_tvm_vm(
        mod,
        opt_level,
        tvm_target,
        params,
        dev,
        tuning_log=tuning_logs,
        tuning_type=tuning_type,
      )
  m.set_input("main", **tvm_inputs)
  tvm_runner = partial(m.invoke, "main")
  benchmark_test(tvm_runner, framework_name = "TVM")

def tvm_profile(
      onnx_model,
      inputs,
      target,
      target_host,
      opt_level = 3,
      freeze=True,
      tuning_logs="",
      use_meta=False,
      use_dnnl=False,
      prune=True,
      model_name="",
    ):
  print("----- TVM profiling of", model_name, "-----")
  mod, params = get_distilbert_mod_params_with_inputs(
                  onnx_model,
                  inputs,
                  opt_level,
                  freeze=freeze,
                  dnnl_enabled=use_dnnl,
                  prune_subgraphs=prune,
                )

  tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in inputs.items()}

  dev = tvm.device(str(target), 0)
  tvm_target = tvm.target.Target(target, target_host)
  if use_meta:
    tuning_type = META_TYPE
  else:
    tuning_type = ANSOR_TYPE
  lib = get_tvm_tuned_vm_lib(
          mod,
          opt_level,
          tvm_target,
          params,
          tuning_log=tuning_logs,
          tuning_type=tuning_type,
        )
  vm = profiler_vm.VirtualMachineProfiler(lib, dev)
  res = vm.profile(func_name="main", **tvm_inputs)
  print("Profiling result:", res)

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
      opt_level,
      database,
      work_dir,
      trials_num = 20000,
      max_trials_per_task=128,
      extracted_task_indices=[],
      excluded_task_indices=[],
    ):
  tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
    extracted_tasks=ms.relay_integration.extract_tasks(
                      mod,
                      target,
                      params,
                      opt_level=opt_level,
                      module_equality=MODULE_EQUALITY),
    work_dir=work_dir,
    space=TUNE_SPACE,
    strategy=TUNE_STRATEGY,
    seed=TUNE_SEED,
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

  if excluded_task_indices:
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
      num_trials_per_iter=TUNE_NUM_TRIALS_PER_ITER,
      builder=TUNE_BUILDER,
      runner=TUNE_RUNNER,
      database=database,
      cost_model=TUNE_COST_MODEL,
      measure_callbacks=TUNE_MEASURE_CALLBACKS,
      task_scheduler=TUNE_TASK_SCHEDULER,
      module_equality=MODULE_EQUALITY,
  )

def tvm_meta_tuning(
      mod,
      params,
      target,
      opt_level,
      trials_num,
      trials_per_task_num,
      log_dir,
      task_indices=[],
      exl_task_indices=[],
    ):
  # Without this, the same workloads with different constant weights
  # are treated as distinct tuning tasks.
  database = get_json_database(log_dir)

  print("Begin meta-scheduler tuning...")
  tune_relay_with_task_extractor(
    mod=mod,
    target=target,
    params=params,
    opt_level=opt_level,
    database=database,
    work_dir=get_work_dir(log_dir),
    trials_num=trials_num,
    max_trials_per_task=trials_per_task_num,
    extracted_task_indices=task_indices,
    excluded_task_indices=exl_task_indices,
  )
  print("Tuning finished")
  pass
