from pathlib import Path
from typing import Union

from tvm import meta_schedule as ms


MODULE_EQUALITY="ignore-ndarray"
# Tuning params for upstreaming relay and TIR
TUNE_SPACE = "post-order-apply"
TUNE_STRATEGY = "replay-trace",  # TODO(vvchernov): "evolutionary",
TUNE_SEED = None
TUNE_BUILDER = "local"
TUNE_RUNNER = "local"
TUNE_COST_MODEL = "xgb"
TUNE_MEASURE_CALLBACKS = "default"
TUNE_TASK_SCHEDULER = "gradient"
TUNE_NUM_TRIALS_PER_ITER = 64

def get_workload_path(dir: Union[str, Path]):
  if isinstance(dir, str):
    dir = Path(dir)
  return str(dir.joinpath("workload.json"))

def get_record_path(dir: Union[str, Path]):
  if isinstance(dir, str):
    dir = Path(dir)
  return str(dir.joinpath("records.json"))

def get_work_dir(dir: Union[str, Path]):
  if isinstance(dir, str):
    dir = Path(dir)
  work_dir = dir.joinpath("work")
  work_dir.mkdir(parents=True, exist_ok=True)
  return str(work_dir)

def get_json_database(db_dir: Union[str, Path]):
  workload_path = get_workload_path(db_dir)
  record_path = get_record_path(db_dir)
  work_dir = get_work_dir(db_dir)

  return ms.database.JSONDatabase(
    workload_path,
    record_path,
    work_dir=work_dir,
    module_equality=MODULE_EQUALITY,
  )
