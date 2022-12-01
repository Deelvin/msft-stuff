from pathlib import Path


MODULE_EQUALITY="ignore-ndarray"

def get_workload_path(dir: Path, name: str):
  return str(dir.joinpath( name + "_workload.json"))

def get_record_path(dir: Path, name: str):
  return str(dir.joinpath( name + "_records.json"))
