from pathlib import Path
from typing import Union


MODULE_EQUALITY="ignore-ndarray"

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
