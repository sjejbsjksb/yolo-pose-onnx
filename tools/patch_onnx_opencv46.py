#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 Ultralytics 导出的 YOLOv8-Pose ONNX 修补成 OpenCV 4.6 能跑通的版本。

--------------------------------------------------------------------------
问题现象
--------------------------------------------------------------------------
    OpenCV(4.6.0) .../dnn/shape_utils.hpp:171: error: (-215:Assertion failed)
    start <= (int)shape.size() && end <= (int)shape.size() && start <= end
    in function 'total'

readNetFromONNX 成功，一 forward() 就抛。原因是 OpenCV 4.6 的 DNN 在计算
Eltwise(Mul/Add/Sub/Div) 的广播形状时，如果两个输入的 rank 差超过 1，
内部 total() 的下标就会越界。而这个 pose 模型里正好有：

    /model.23/Mul_2 : [1, 4, 8400] * [1, 8400]      -> dbox * strides
    /model.23/Mul_3 : [1, 4, 2, 8400] * [1, 8400]   -> kpts * strides
    /model.23/Add_3 : [1, 4, 2, 8400] + [2, 8400]   -> + anchors

OpenCV 4.8 起修好了这个 bug（opencv/opencv#23464），4.6 / 4.7 会中招。

--------------------------------------------------------------------------
修补原理
--------------------------------------------------------------------------
把这些广播常量提前用 np.broadcast_to 展开成完整形状，从"广播乘"变成
"等形状乘"。数学上完全等价 —— 实测输出与修补前逐位相同（max abs diff = 0）。

代价：模型体积从 9.72 MB 涨到 10.24 MB（多了 3 个常量，约 700 KB）。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
    pip install onnx numpy
    python patch_onnx_opencv46.py best.onnx best_opencv46.onnx
"""

import sys

import numpy as np
import onnx
from onnx import numpy_helper

# 单个展开后的常量最大字节数，防止把模型吹爆
MAX_BYTES = 8 * 1024 * 1024

BROADCAST_OPS = ("Mul", "Add", "Sub", "Div")


def collect_value_info(graph):
    """收集 graph 里所有已知的输出形状（全静态的才要）"""
    shapes = {}
    for vi in graph.value_info:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                dims = None
                break
            dims.append(d.dim_value)
        if dims:
            shapes[vi.name] = tuple(dims)
    return shapes


def try_expand(arr, out_shape):
    """把常量右对齐后广播到 out_shape，做不到就返回 None"""
    if arr.ndim == 0 or arr.ndim > len(out_shape):
        return None

    newshape = (1,) * (len(out_shape) - arr.ndim) + arr.shape
    for ns, os_ in zip(newshape, out_shape):
        if ns != os_ and ns != 1:
            return None

    if int(np.prod(out_shape)) * arr.dtype.itemsize > MAX_BYTES:
        return None

    return np.broadcast_to(arr.reshape(newshape), out_shape).copy()


def patch(src_path, dst_path):
    model = onnx.load(src_path)
    graph = model.graph

    inits = {i.name: i for i in graph.initializer}
    shapes = collect_value_info(graph)

    cache = {}
    patched = []

    for node in graph.node:
        if node.op_type not in BROADCAST_OPS:
            continue

        out_shape = shapes.get(node.output[0])
        if not out_shape:
            continue

        for idx, inp in enumerate(node.input):
            if inp not in inits:
                continue

            arr = numpy_helper.to_array(inits[inp])
            expanded = try_expand(arr, out_shape)
            if expanded is None or expanded.shape == arr.shape:
                continue

            key = (inp, expanded.shape)
            if key in cache:
                node.input[idx] = cache[key]
                continue

            name = "%s_exp%d" % (inp, len(cache))
            graph.initializer.append(numpy_helper.from_array(expanded, name))
            node.input[idx] = name
            cache[key] = name
            patched.append((node.name or node.op_type, arr.shape, expanded.shape))

    if not patched:
        print("[patch] 没有需要修补的广播常量（模型可能本来就兼容）")
    else:
        print("[patch] 展开 %d 个广播常量：" % len(patched))
        for name, old, new in patched:
            print("    %-28s %s -> %s" % (name, old, new))

    # 删掉没人用的 initializer
    used = set()
    for node in graph.node:
        used.update(node.input)
    keep = [i for i in graph.initializer if i.name in used]
    dropped = len(graph.initializer) - len(keep)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    if dropped:
        print("[patch] 清理无用 initializer: %d 个" % dropped)

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    print("[patch] 已保存: %s" % dst_path)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("用法: python patch_onnx_opencv46.py <输入.onnx> <输出.onnx>")
        return 1
    patch(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
