# Default TIR of fused_nn_dense_multiply_subtract_cast_multiply_cast_multiply_subtract_expand_dim_f2f16df33f319036__2
# It is first in vm_profiler table
import numpy as np

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class ModuleD1:
  @T.prim_func
  def main(p0: T.Buffer[(T.int64(54), T.int64(3072)), "uint8"],
           p1: T.Buffer[(T.int64(768), T.int64(3072)), "int8"],
           p2: T.Buffer[(T.int64(54), T.int64(1)), "int32"],
           p3: T.Buffer[(), "uint8"],
           p4: T.Buffer[T.int64(768), "int32"],
           T_add: T.Buffer[(T.int64(54), T.int64(768)), "int32"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # buffer definition
    p1_1 = T.buffer_decl([768, 3072], dtype="int8", data=p1.data)
    # body
    # with T.block("root")
    T_matmul_NT = T.alloc_buffer([T.int64(54), T.int64(768)], dtype="int32")
    compile_engine_const = T.alloc_buffer([], dtype="int32")
    T_multiply = T.alloc_buffer([T.int64(54), T.int64(1)], dtype="int32")
    T_subtract = T.alloc_buffer([T.int64(54), T.int64(768)], dtype="int32")
    T_cast = T.alloc_buffer([], dtype="int32")
    compile_engine_const_1 = T.alloc_buffer([], dtype="int32")
    T_multiply_1 = T.alloc_buffer([], dtype="int32")
    T_cast_1 = T.alloc_buffer([], dtype="int32")
    T_multiply_2 = T.alloc_buffer([T.int64(768)], dtype="int32")
    T_subtract_1 = T.alloc_buffer([T.int64(768)], dtype="int32")
    T_expand_dims = T.alloc_buffer([T.int64(1), T.int64(768)], dtype="int32")
    for i, j, k in T.grid(T.int64(54), T.int64(768), T.int64(3072)):
      with T.block("T_matmul_NT"):
        v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
        T.reads(p0[v_i, v_k], p1[v_j, v_k])
        T.writes(T_matmul_NT[v_i, v_j])
        T.block_attr({"layout_free_placeholders":[p1_1]})
        with T.init():
          T_matmul_NT[v_i, v_j] = 0
        T_matmul_NT[v_i, v_j] = T_matmul_NT[v_i, v_j] + T.Cast("int32", p0[v_i, v_k]) * T.Cast("int32", p1[v_j, v_k])
    with T.block("compile_engine_const"):
      vi = T.axis.spatial(T.int64(1), T.int64(0))
      T.reads()
      T.writes(compile_engine_const[()])
      compile_engine_const[()] = 0
    for ax0, ax1 in T.grid(T.int64(54), T.int64(1)):
      with T.block("T_multiply"):
        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
        T.reads(compile_engine_const[()], p2[v_ax0, v_ax1])
        T.writes(T_multiply[v_ax0, v_ax1])
        T_multiply[v_ax0, v_ax1] = compile_engine_const[()] * p2[v_ax0, v_ax1]
    for ax0, ax1 in T.grid(T.int64(54), T.int64(768)):
      with T.block("T_subtract"):
        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
        T.reads(T_matmul_NT[v_ax0, v_ax1], T_multiply[v_ax0, T.int64(0)])
        T.writes(T_subtract[v_ax0, v_ax1])
        T_subtract[v_ax0, v_ax1] = T_matmul_NT[v_ax0, v_ax1] - T_multiply[v_ax0, T.int64(0)]
    with T.block("T_cast"):
      vi = T.axis.spatial(T.int64(1), T.int64(0))
      T.reads(p3[()])
      T.writes(T_cast[()])
      T_cast[()] = T.Cast("int32", p3[()])
    with T.block("compile_engine_const_1"):
      vi = T.axis.spatial(T.int64(1), T.int64(0))
      T.reads()
      T.writes(compile_engine_const_1[()])
      compile_engine_const_1[()] = 0
    with T.block("T_multiply_1"):
      vi = T.axis.spatial(T.int64(1), T.int64(0))
      T.reads(T_cast[()], compile_engine_const_1[()])
      T.writes(T_multiply_1[()])
      T_multiply_1[()] = T_cast[()] * compile_engine_const_1[()]
    with T.block("T_cast_1"):
      vi = T.axis.spatial(T.int64(1), T.int64(0))
      T.reads(p3[()])
      T.writes(T_cast_1[()])
      T_cast_1[()] = T.Cast("int32", p3[()])
    for ax0 in T.serial(T.int64(768)):
      with T.block("T_multiply_2"):
        v_ax0 = T.axis.spatial(T.int64(768), ax0)
        T.reads(T_cast_1[()], p4[v_ax0])
        T.writes(T_multiply_2[v_ax0])
        T_multiply_2[v_ax0] = T_cast_1[()] * p4[v_ax0]
    for ax0 in T.serial(T.int64(768)):
      with T.block("T_subtract_1"):
        v_ax0 = T.axis.spatial(T.int64(768), ax0)
        T.reads(T_multiply_1[()], T_multiply_2[v_ax0])
        T.writes(T_subtract_1[v_ax0])
        T_subtract_1[v_ax0] = T_multiply_1[()] - T_multiply_2[v_ax0]
    for ax0, ax1 in T.grid(T.int64(1), T.int64(768)):
      with T.block("T_expand_dims"):
        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
        T.reads(T_subtract_1[v_ax1])
        T.writes(T_expand_dims[v_ax0, v_ax1])
        T_expand_dims[v_ax0, v_ax1] = T_subtract_1[v_ax1]
    for ax0, ax1 in T.grid(T.int64(54), T.int64(768)):
      with T.block("T_add"):
        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
        T.reads(T_subtract[v_ax0, v_ax1], T_expand_dims[T.int64(0), v_ax1])
        T.writes(T_add[v_ax0, v_ax1])
        T_add[v_ax0, v_ax1] = T_subtract[v_ax0, v_ax1] + T_expand_dims[T.int64(0), v_ax1]

  def get_rnd_inputs():
    p0_shape = (54, 3072,)
    p1_shape = (768, 3072,)
    p2_shape = (54, 1,)
    p3_shape = ()
    p4_shape = (768,)
    Tadd_shape = (54, 768,)

    p0_np = np.random.randn(*p0_shape).astype("uint8")
    p1_np = np.random.randn(*p1_shape).astype("int8")
    p2_np = np.random.randn(*p2_shape).astype("int32")
    p3_np = np.random.randn(*p3_shape).astype("uint8")
    p4_np = np.random.randn(*p4_shape).astype("int32")
    Tadd_np = np.random.randn(*Tadd_shape).astype("int32")

    p0_nd = tvm.nd.array(p0_np)
    p1_nd = tvm.nd.array(p1_np)
    p2_nd = tvm.nd.array(p2_np)
    p3_nd = tvm.nd.array(p3_np)
    p4_nd = tvm.nd.array(p4_np)
    Tadd_nd = tvm.nd.array(Tadd_np)

    return [p0_nd, p1_nd, p2_nd, p3_nd, p4_nd, Tadd_nd,]
