from pathlib import Path
from typing import Union


MODULE_EQUALITY="ignore-ndarray"

def get_workload_path(dir: Union[str, Path], name: str):
  if isinstance(dir, str):
    dir = Path(dir)
  return str(dir.joinpath( name + "_workload.json"))

def get_record_path(dir: Union[str, Path], name: str):
  if isinstance(dir, str):
    dir = Path(dir)
  return str(dir.joinpath( name + "_records.json"))
