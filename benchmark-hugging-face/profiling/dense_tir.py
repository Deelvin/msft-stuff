# Dense TIR from fused_nn_dense_multiply_subtract_cast_multiply_cast_multiply_subtract_expand_dim_f2f16df33f319036__2
# It is fused kernel from default_tir_2.py
import numpy as np

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class ModuleDense:
  @T.prim_func
  def main(p0: T.Buffer[(T.int64(54), T.int64(3072)), "uint8"],
           p1: T.Buffer[(T.int64(768), T.int64(3072)), "int8"],
           T_matmul: T.Buffer[(T.int64(54), T.int64(768)), "int32"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # buffer definition
    p1_1 = T.buffer_decl([768, 3072], dtype="int8", data=p1.data)
    for i, j, k in T.grid(T.int64(54), T.int64(768), T.int64(3072)):
      with T.block("T_matmul"):
        v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
        T.reads(p0[v_i, v_k], p1[v_j, v_k])
        T.writes(T_matmul[v_i, v_j])
        T.block_attr({"layout_free_placeholders":[p1_1]})
        with T.init():
          T_matmul[v_i, v_j] = 0
        T_matmul[v_i, v_j] = T_matmul[v_i, v_j] + T.Cast("int32", p0[v_i, v_k]) * T.Cast("int32", p1[v_j, v_k])


def get_rnd_inputs():
  p0_shape = (54, 3072,)
  p1_shape = (768, 3072,)
  T_matmul_shape = (54, 768,)

  p0_np = np.random.randn(*p0_shape).astype("uint8")
  p1_np = np.random.randn(*p1_shape).astype("int8")
  T_matmul_np = np.random.zero(*T_matmul_shape).astype("int32")

  p0_nd = tvm.nd.array(p0_np)
  p1_nd = tvm.nd.array(p1_np)
  T_matmul_nd = tvm.nd.array(T_matmul_np)

  return [p0_nd, p1_nd, T_matmul_nd,]
