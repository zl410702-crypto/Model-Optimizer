#!/usr/bin/env python3
"""用 ModelOpt 对现有 ViT 做 INT8 PTQ，并导出包含 ONNX Q/DQ 的模型。"""

from __future__ import annotations

import argparse
import copy
import contextlib
from collections import Counter
import json
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import onnx
import timm
import torch
from onnx import TensorProto, numpy_helper
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ONNX = REPO_ROOT / "examples/onnx_ptq/work_vit/vit_base_patch16_224.onnx"
DEFAULT_CALIBRATION = REPO_ROOT / "examples/onnx_ptq/work_vit/calib64.npy"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/vit_modelopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, default=DEFAULT_SOURCE_ONNX)
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "fp32_onnx": output_dir / "vit_fp32.onnx",
        "int8_onnx": output_dir / "vit_int8_qdq.onnx",
        "config": output_dir / "quant_config.json",
        "quant_summary": output_dir / "quant_summary.txt",
        "onnx_summary": output_dir / "onnx_summary.txt",
        "fp32_summary": output_dir / "onnx_fp32_summary.txt",
    }


def load_vit_model(source_onnx: Path, device: torch.device) -> nn.Module:
    """建立 timm ViT，并从已有浮点 ONNX 读回全部权重，不下载或改写权重文件。"""
    source = onnx.load(str(source_onnx), load_external_data=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)

    # 关闭 fused SDPA，让 opset 17 导出为基础 MatMul/Softmax 图，便于旧版 TRT parser 解析。
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False

    onnx_weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in source.graph.initializer}
    state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}

    # 大多数参数保留 PyTorch 参数名和维度，可以直接映射。
    for name, value in state.items():
        array = onnx_weights.get(name)
        if array is not None and tuple(array.shape) == tuple(value.shape):
            loaded[name] = torch.from_numpy(np.array(array, copy=True)).to(dtype=value.dtype)

    # ONNX 将 Linear 写成 X @ W；PyTorch Linear 保存 [out,in]，所以从 ONNX 转置回来。
    for node in source.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2 or node.input[1] not in onnx_weights:
            continue
        parts = node.name.strip("/").split("/")
        if len(parts) < 4 or parts[0] != "blocks":
            continue
        module_name = ".".join(parts[1:-1])
        weight_name = f"{module_name}.weight"
        if weight_name not in state:
            continue
        array = onnx_weights[node.input[1]].T
        if tuple(array.shape) == tuple(state[weight_name].shape):
            loaded[weight_name] = torch.from_numpy(np.array(array, copy=True)).to(
                dtype=state[weight_name].dtype
            )

    missing = sorted(set(state) - set(loaded))
    if missing:
        raise RuntimeError(
            "无法从来源 ONNX 映射全部 ViT 参数；未加载参数示例：" + ", ".join(missing[:8])
        )
    model.load_state_dict(loaded, strict=True)
    return model.eval().to(device)


def prepare_calibration_data(path: Path, sample_count: int, device: torch.device) -> torch.Tensor:
    """读取仓库已有的校准张量；其原始数据集来源没有随文件保存。"""
    array = np.load(path)
    if array.ndim != 4 or tuple(array.shape[1:]) != (3, 224, 224):
        raise ValueError(f"校准数据应为 [N,3,224,224]，实际为 {array.shape}")
    if array.shape[0] < sample_count:
        raise ValueError(f"请求 {sample_count} 个样本，但数据只有 {array.shape[0]} 个")
    return torch.from_numpy(np.asarray(array[:sample_count], dtype=np.float32).copy()).to(device)


def configure_quantizers() -> dict:
    """复制 ModelOpt 0.41.0 官方 INT8 默认配置，避免改动全局常量。"""
    return copy.deepcopy(mtq.INT8_DEFAULT_CFG)


def write_quant_summary(model: nn.Module, path: Path) -> None:
    """保存官方 quant summary，并列出各 quantizer 的 amax、scale 和 axis。"""
    with path.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
        print("ModelOpt mtq.print_quant_summary():")
        mtq.print_quant_summary(model)
        print("\nQuantizer amax / scale / axis:")
        for name, module in model.named_modules():
            if not isinstance(module, TensorQuantizer):
                continue
            amax = module.amax
            if amax is None:
                print(f"{name}: amax 未设置")
                continue
            amax_cpu = amax.detach().float().cpu()
            scale = amax_cpu / float(module.maxbound)
            axis = module.axis
            print(
                f"{name}: bits={module.num_bits}, axis={axis}, "
                f"amax_shape={tuple(amax_cpu.shape)}, "
                f"amax_range=({amax_cpu.min().item():.8g},{amax_cpu.max().item():.8g}), "
                f"scale_shape={tuple(scale.shape)}, "
                f"scale_range=({scale.min().item():.8g},{scale.max().item():.8g}), "
                f"scale_first={scale.reshape(-1)[:5].tolist()}"
            )


def _constant_value(name: str, producers: dict[str, onnx.NodeProto], initializers: dict[str, np.ndarray]):
    if name in initializers:
        return initializers[name]
    node = producers.get(name)
    if node is None:
        return None
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
            if attr.name == "value_int":
                return np.asarray(attr.i, dtype=np.int64)
            if attr.name == "value_float":
                return np.asarray(attr.f, dtype=np.float32)
    if node.op_type == "Cast" and node.input:
        value = _constant_value(node.input[0], producers, initializers)
        target = next((attr.i for attr in node.attribute if attr.name == "to"), None)
        if value is not None and target is not None:
            return value.astype(onnx.helper.tensor_dtype_to_np_dtype(target))
        return value
    return None


def _find_weight_qdq(value: str, producers: dict[str, onnx.NodeProto], initializers: dict[str, np.ndarray], depth=0):
    if depth > 8:
        return None
    node = producers.get(value)
    if node is None:
        return None
    if node.op_type == "DequantizeLinear" and node.input:
        q_node = producers.get(node.input[0])
        if q_node is not None and q_node.op_type == "QuantizeLinear":
            source = q_node.input[0]
            if source in initializers:
                return source, q_node, node
        return None
    # 只穿过权重布局算子，避免沿激活的计算图误追到更早层的权重量化。
    if node.op_type in {"Transpose", "Identity", "Reshape", "Cast"} and node.input:
        return _find_weight_qdq(node.input[0], producers, initializers, depth + 1)
    return None


def validate_and_summarize_onnx(path: Path, summary_path: Path, require_qdq: bool = True) -> None:
    """运行 ONNX Checker 并统计 Q/DQ 参数和线性层连接。"""
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    counts = Counter(node.op_type for node in graph.node)
    quantizers = [node for node in graph.node if node.op_type == "QuantizeLinear"]
    dequantizers = [node for node in graph.node if node.op_type == "DequantizeLinear"]

    qdq_specs = Counter()
    quant_stats: dict[str, dict] = {}
    for node in quantizers:
        scale = _constant_value(node.input[1], producers, initializers) if len(node.input) > 1 else None
        zero = _constant_value(node.input[2], producers, initializers) if len(node.input) > 2 else None
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
        scale_info = (tuple(scale.shape), str(scale.dtype)) if scale is not None else (None, None)
        zero_info = (tuple(zero.shape), str(zero.dtype)) if zero is not None else (None, None)
        qdq_specs[(axis, scale_info, zero_info)] += 1
        role = "weight" if node.input[0] in initializers else "activation"
        bucket = quant_stats.setdefault(
            role,
            {"axes": Counter(), "scale_shapes": Counter(), "scale_values": [], "zero_dtypes": Counter(), "zero_values": set()},
        )
        bucket["axes"][axis] += 1
        if scale is not None:
            bucket["scale_shapes"][tuple(scale.shape)] += 1
            bucket["scale_values"].extend(np.asarray(scale).reshape(-1).tolist())
        if zero is not None:
            bucket["zero_dtypes"][str(zero.dtype)] += 1
            bucket["zero_values"].update(np.asarray(zero).reshape(-1).tolist())

    linear_nodes = [node for node in graph.node if node.op_type in {"MatMul", "Gemm"}]
    weight_qdq = 0
    activation_qdq = 0
    examples = []
    for node in linear_nodes:
        weight_path = _find_weight_qdq(node.input[1], producers, initializers) if len(node.input) > 1 else None
        activation_parent = producers.get(node.input[0]) if node.input else None
        activation_has_dq = activation_parent is not None and activation_parent.op_type == "DequantizeLinear"
        if weight_path:
            weight_qdq += 1
        if activation_has_dq:
            activation_qdq += 1
        if len(examples) < 8:
            examples.append(
                f"{node.name or node.op_type}: activation_input_DQ={activation_has_dq}; "
                f"weight_input_QDQ={weight_path is not None}"
            )

    if require_qdq and (not quantizers or not dequantizers):
        raise RuntimeError(f"{path} 没有同时包含 QuantizeLinear 和 DequantizeLinear")

    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("ONNX Checker: PASS\n")
        stream.write(f"Producer: {model.producer_name} {model.producer_version}\n")
        stream.write(f"IR version: {model.ir_version}\n")
        stream.write(f"Opset: {[(op.domain or 'ai.onnx', op.version) for op in model.opset_import]}\n")
        stream.write(f"Input: {[(x.name, TensorProto.DataType.Name(x.type.tensor_type.elem_type), [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in graph.input]}\n")
        stream.write(f"Output: {[(x.name, TensorProto.DataType.Name(x.type.tensor_type.elem_type), [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in graph.output]}\n")
        stream.write(f"Node count: {len(graph.node)}\n")
        stream.write(f"QuantizeLinear: {len(quantizers)}\nDequantizeLinear: {len(dequantizers)}\n")
        stream.write(f"Q scale/zero-point/axis distributions: {dict(qdq_specs)}\n")
        stream.write("Q/DQ numerical summary by source tensor class:\n")
        for role, bucket in quant_stats.items():
            scales = bucket["scale_values"]
            stream.write(
                f"  {role}: count={sum(bucket['axes'].values())}, axes={dict(bucket['axes'])}, "
                f"scale_shapes={dict(bucket['scale_shapes'])}, "
                f"scale_dtype=FLOAT, scale_range=({min(scales):.9g},{max(scales):.9g}), "
                f"zero_point_dtype={dict(bucket['zero_dtypes'])}, "
                f"zero_point_values={sorted(bucket['zero_values'])}\n"
            )
        stream.write("Q output dtype: INT8 (from signed INT8 zero point); DQ output dtype: FLOAT\n")
        stream.write(f"MatMul/Gemm count: {len(linear_nodes)}\n")
        stream.write(f"Linear input activations with direct DQ: {activation_qdq}\n")
        stream.write(f"Linear weights with traceable initializer->Q->DQ path: {weight_qdq}\n")
        stream.write("\nOperator counts:\n")
        for op, count in counts.most_common():
            stream.write(f"{op}: {count}\n")
        stream.write("\nMatMul/Gemm Q/DQ examples:\n")
        for example in examples:
            stream.write(example + "\n")


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("为避免覆盖，以下输出文件已存在：\n" + "\n".join(existing))
    if not args.source_onnx.is_file() or not args.calibration_data.is_file():
        raise FileNotFoundError("来源 ONNX 或 Calibration 数据不存在")
    if args.opset < 13 or args.opset > 17:
        raise ValueError("为兼容 TensorRT 8.6.x，本脚本使用 opset 13 到 17；默认 opset 17。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch={torch.__version__}; ModelOpt={package_version('nvidia-modelopt')}")
    print(f"Device={device}; ONNX={onnx.__version__}; opset={args.opset}")

    # 1. Load ViT Model：从已存在的 ONNX 参数恢复 timm ViT，不访问网络。
    model = load_vit_model(args.source_onnx, device)
    fp32_model = copy.deepcopy(model).eval()
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)

    # 2. Configure ModelOpt Quantizers：INT8 权重 per-channel，激活 per-tensor。
    quant_config = configure_quantizers()
    with paths["config"].open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "modelopt_version": package_version("nvidia-modelopt"),
                "source_weights": str(args.source_onnx),
                "pretrained_provenance": "沿用来源 ONNX 参数；该 ONNX 未记录可验证的预训练 checkpoint 来源。",
                "calibration_data": str(args.calibration_data),
                "calibration_provenance": "复用现有 calib64.npy；文件没有数据集来源元信息，无法核实样本来源。",
                "quant_config": quant_config,
                "calibration_samples": args.calibration_samples,
                "calibration_batch_size": args.calibration_batch_size,
                "onnx_opset": args.opset,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    # 3. Prepare Calibration Data：复用仓库数组，不在脚本中生成随机校准输入。
    calibration = prepare_calibration_data(args.calibration_data, args.calibration_samples, device)

    # 4. Run ModelOpt PTQ：forward_loop 让输入经过模型，以收集激活 amax。
    def forward_loop(calibration_model: nn.Module) -> None:
        calibration_model.eval()
        with torch.inference_mode():
            for start in range(0, calibration.shape[0], args.calibration_batch_size):
                batch = calibration[start : start + args.calibration_batch_size]
                calibration_model(batch)

    quantized_model = mtq.quantize(model, quant_config, forward_loop)
    quantized_model.eval()
    write_quant_summary(quantized_model, paths["quant_summary"])

    # 5. Export ONNX：同一组基础权重分别导出浮点图与带 Q/DQ 的 INT8 图。
    # 本机 ModelOpt CUDA extension 未安装；GPU 上调用 legacy exporter 会崩溃。
    # 将已校准模型搬到 CPU 仅用于 tracing，不改变权重、amax 或导出量化语义。
    fp32_model = fp32_model.cpu()
    quantized_model = quantized_model.cpu()
    export_input = dummy_input.cpu()
    with torch.inference_mode():
        torch.onnx.export(
            fp32_model,
            export_input,
            str(paths["fp32_onnx"]),
            input_names=["images"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        torch.onnx.export(
            quantized_model,
            export_input,
            str(paths["int8_onnx"]),
            input_names=["images"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # 6. Validate ONNX Q/DQ：Checker、Q/DQ 参数和 Linear 连接统计。
    validate_and_summarize_onnx(paths["fp32_onnx"], paths["fp32_summary"], require_qdq=False)
    validate_and_summarize_onnx(paths["int8_onnx"], paths["onnx_summary"])
    if paths["onnx_summary"].stat().st_size == 0:
        raise RuntimeError("ONNX summary 未生成")
    print("完成：")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"fp32_summary: {paths['fp32_summary']}")


if __name__ == "__main__":
    main()
