# Simple TIR for add operation, need for testing
import numpy as np

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class BaseAdd:
  @T.prim_func
  def main(A: T.Buffer[(4, 4), "int64"],
           B: T.Buffer[(4, 4), "int64"],
           C: T.Buffer[(4, 4), "int64"]
      ):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

def get_rnd_inputs():
  a = np.arange(16).reshape(4, 4)
  b = np.arange(16, 0, -1).reshape(4, 4)

  a_tvm = tvm.nd.array(a)
  b_tvm = tvm.nd.array(b)
  c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))

  return [a_tvm, b_tvm, c_tvm,]
