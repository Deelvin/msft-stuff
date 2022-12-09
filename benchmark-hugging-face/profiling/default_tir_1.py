# Default TIR of fused_divide_round_cast_cast_add_clip_cast_reshape__3
# It is first in vm_profiler table
import numpy as np

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class ModuleD1:
  @T.prim_func
  def main(p0: T.Buffer[(T.int64(1), T.int64(54), T.int64(3072)), "float32"],
           p1: T.Buffer[(), "float32"],
           p2: T.Buffer[(), "uint8"],
           T_reshape: T.Buffer[(T.int64(54), T.int64(3072)), "uint8"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    T_divide = T.alloc_buffer([T.int64(1), T.int64(54), T.int64(3072)], dtype="float32")
    T_round = T.alloc_buffer([T.int64(1), T.int64(54), T.int64(3072)], dtype="float32")
    T_cast = T.alloc_buffer([], dtype="int32")
    T_cast_1 = T.alloc_buffer([], dtype="float32")
    T_add = T.alloc_buffer([T.int64(1), T.int64(54), T.int64(3072)], dtype="float32")
    compute = T.alloc_buffer([T.int64(1), T.int64(54), T.int64(3072)], dtype="float32")
    T_cast_2 = T.alloc_buffer([T.int64(1), T.int64(54), T.int64(3072)], dtype="uint8")
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(54), T.int64(3072)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(p0[v_ax0, v_ax1, v_ax2], p1[()])
            T.writes(T_divide[v_ax0, v_ax1, v_ax2])
            T_divide[v_ax0, v_ax1, v_ax2] = p0[v_ax0, v_ax1, v_ax2] / p1[()]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(54), T.int64(3072)):
        with T.block("T_round"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_divide[v_ax0, v_ax1, v_ax2])
            T.writes(T_round[v_ax0, v_ax1, v_ax2])
            T_round[v_ax0, v_ax1, v_ax2] = T.round(T_divide[v_ax0, v_ax1, v_ax2], dtype="float32")
    with T.block("T_cast"):
        vi = T.axis.spatial(T.int64(1), T.int64(0))
        T.reads(p2[()])
        T.writes(T_cast[()])
        T_cast[()] = T.Cast("int32", p2[()])
    with T.block("T_cast_1"):
        vi = T.axis.spatial(T.int64(1), T.int64(0))
        T.reads(T_cast[()])
        T.writes(T_cast_1[()])
        T_cast_1[()] = T.Cast("float32", T_cast[()])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(54), T.int64(3072)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_round[v_ax0, v_ax1, v_ax2], T_cast_1[()])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T_round[v_ax0, v_ax1, v_ax2] + T_cast_1[()]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(54), T.int64(3072)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_add[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.max(T.min(T_add[v_i0, v_i1, v_i2], T.float32(255)), T.float32(0))
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(54), T.int64(3072)):
        with T.block("T_cast_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_cast_2[v_ax0, v_ax1, v_ax2])
            T_cast_2[v_ax0, v_ax1, v_ax2] = T.Cast("uint8", compute[v_ax0, v_ax1, v_ax2])
    for ax0, ax1 in T.grid(T.int64(54), T.int64(3072)):
        with T.block("T_reshape"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(T_cast_2[T.int64(0), (v_ax1 // T.int64(3072) + v_ax0) % T.int64(54), v_ax1 % T.int64(3072)])
            T.writes(T_reshape[v_ax0, v_ax1])
            T_reshape[v_ax0, v_ax1] = T_cast_2[T.int64(0), (v_ax1 // T.int64(3072) + v_ax0) % T.int64(54), v_ax1 % T.int64(3072)]

def get_rnd_inputs():
  p0_shape = (1, 54, 3072,)
  # p1_shape = () scalar
  # p2_shape = () scalar
  Treshape_shape = (54, 3072,)

  p0_np = np.random.randn(*p0_shape).astype("float32")
  p1_np = np.random.rand()
  p2_np = np.random.randint(0, 255, dtype="uint8")
  Treshape_np = np.random.randn(*Treshape_shape).astype("uint8")

  p0_nd = tvm.nd.array(p0_np)
  p1_nd = tvm.nd.array(p1_np)
  p2_nd = tvm.nd.array(p2_np)
  Treshape_nd = tvm.nd.array(Treshape_np)

  return [p0_nd, p1_nd, p2_nd, Treshape_nd,]
