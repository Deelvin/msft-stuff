import os


def project_root() -> str:
    return os.path.sep.join(os.path.abspath(__file__).split(sep=os.path.sep)[:-2])
