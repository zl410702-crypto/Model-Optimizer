# ViT ModelOpt INT8 PTQ → ONNX Q/DQ

## 已验证结果

本流程在当前环境实际运行成功：PyTorch `2.8.0+cu128`、ModelOpt `0.41.0`、ONNX `1.19.1`、timm `1.0.30`，使用宿主机 RTX 4060 和 64 个现有校准张量完成 PTQ。没有训练模型，也没有改写来源 ONNX。

来源权重从 `examples/onnx_ptq/work_vit/vit_base_patch16_224.onnx` 读入 timm 架构。代码用 `pretrained=False` 建立模型，再将 ONNX 中全部 152 个参数映射回模型（Linear 权重按 MatMul 布局转置）。没有下载新 checkpoint；来源 ONNX 没有记录可验证的预训练来源，所以无法确认它是否来自预训练权重。现有 `calib64.npy` 是 `[64,3,224,224]` 的 float32 张量，数值范围 `[-1,1]`；文件没有数据集来源信息。它确实用于校准，但样本来源无法核实，也没有用它评价分类准确率。

脚本将 Q/DQ ONNX 固定导出为 opset 17。ONNX Checker 通过；INT8 图有 **100 个 `QuantizeLinear` 和 100 个 `DequantizeLinear`**。当前系统没有 TensorRT 8.6.4 或 TensorRT Python 包；本机缓存的 TensorRT 8.5.3 parser 对该 opset 17 图仍在 `LayerNormalization` 节点失败。Parser 日志保存在 `tensorrt_8.5.3_parser.txt`。没有构建 engine。

## 运行方式

在仓库根目录启动已安装的 conda 环境：

```bash
source /home/eric/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/modelopt041
python tools/quantize_vit_modelopt.py
```

默认读取：

- `examples/onnx_ptq/work_vit/vit_base_patch16_224.onnx`
- `examples/onnx_ptq/work_vit/calib64.npy`

默认写入 `outputs/vit_modelopt/`。脚本检测到任何目标文件已存在时会停止，防止覆盖；本次任务已生成默认输出，若要再次运行，请给新目录：

```bash
python tools/quantize_vit_modelopt.py --output-dir outputs/vit_modelopt_run2
```

也可指定 `--calibration-samples`、`--calibration-batch-size` 和 `--opset`。脚本接受 opset 13–17，默认 17。模型在 GPU 上做校准；当前 ModelOpt CUDA exporter extension 未安装，实验中 GPU legacy ONNX tracing 会段错误，因此导出前把同一份已校准模型移到 CPU tracing。量化权重、amax 和导出图语义不因此改变。

## 文件说明

| 输出 | 内容 |
|---|---|
| `vit_fp32.onnx` | 来源权重恢复后的浮点 ONNX，opset 17；输入输出 `[1,3,224,224]` / `[1,1000]`。 |
| `vit_int8_qdq.onnx` | ModelOpt Torch quantizer 导出的显式 INT8 Q/DQ ONNX，opset 17。 |
| `quant_config.json` | 实际使用的 ModelOpt 0.41.0 配置、校准样本数、来源路径及来源限制。 |
| `quant_summary.txt` | `mtq.print_quant_summary()` 输出，以及逐个 quantizer 的 `amax`、scale、axis。 |
| `onnx_summary.txt` | ONNX Checker、opset、节点计数、Q/DQ 参数 dtype/shape/axis、MatMul/Gemm 连线统计。 |
| `onnx_fp32_summary.txt` | FP32 ONNX Checker 和 operator 统计；此图没有 Q/DQ。 |
| `tensorrt_8.5.3_parser.txt` | 用缓存中的 8.5.3 parser 仅解析 Q/DQ 图的日志；没有构建 engine。 |

## Python 调用链

脚本按 `Load ViT Model → Configure Quantizer → Calibration → mtq.quantize() → torch.onnx.export() → Validate ONNX` 组织：


1. **Load ViT Model**：[`load_vit_model()` 第 53 行起](../../tools/quantize_vit_modelopt.py#L53) 创建 `timm.create_model("vit_base_patch16_224", pretrained=False)`，然后从现有 ONNX 恢复参数。48 个 Transformer Linear 权重从 ONNX `MatMul` initializer 转置回 PyTorch `[out,in]`；其余按参数名和 shape 对应。全部参数 `strict=True` 加载，避免漏权重后悄悄保留随机初始化。

2. **Configure ModelOpt Quantizers**：[`configure_quantizers()` 第 109 行起](../../tools/quantize_vit_modelopt.py#L109) 深拷贝当前版本的 `mtq.INT8_DEFAULT_CFG`。实际 JSON 中核心配置是：

   ```json
   {
     "*weight_quantizer": {"num_bits": 8, "axis": 0},
     "*input_quantizer": {"num_bits": 8, "axis": null},
     "algorithm": "max"
   }
   ```

   `weight_quantizer` 对每个输出通道分别量化；PyTorch `Linear.weight` 布局为 `[out,in]`，所以 axis 为 0。`input_quantizer` 对整个输入张量用一个 scale。该默认配置把模块 `output_quantizer` 设为 disabled，所以这里导出的 Q/DQ 主要位于 Linear/Conv 输入与权重路径。

3. **Prepare Calibration Data**：[`prepare_calibration_data()` 第 99 行起](../../tools/quantize_vit_modelopt.py#L99) 读取现有 `.npy`，检查 shape，取 64 个样本。数据按 batch size 8 放到 GPU，不生成随机校准输入。

4. **Run ModelOpt PTQ**：[`forward_loop()` 与 `mtq.quantize()` 第 312–320 行](../../tools/quantize_vit_modelopt.py#L312) 中，`forward_loop(calibration_model)` 接收 ModelOpt 传入的模型，并逐 batch 调用 `calibration_model(batch)`。`mtq.quantize(model, quant_config, forward_loop)` 会把可量化 Linear/Conv 转为带 `TensorQuantizer` 的 quantized modules、通过 forward loop 收集激活统计，再按 `algorithm="max"` 固定量化参数；它返回量化后的模型。该 API 会对模型做量化转换，脚本保留独立 `fp32_model` 副本用于导出浮点图。

5. **Inspect Quantizers**：[`write_quant_summary()` 第 114 行起](../../tools/quantize_vit_modelopt.py#L114) 调用 `mtq.print_quant_summary(model)`。随后读取每个 `TensorQuantizer.amax`、`axis` 与 `maxbound`，以 `scale = amax / 127` 输出 scale 范围和前几个值。对 8-bit signed INT8，ONNX zero point 为 0。

6. **Export and Validate ONNX**：[`torch.onnx.export()` 第 329–348 行](../../tools/quantize_vit_modelopt.py#L329) 分别导出 FP32 和 INT8 图，`opset_version=17`。ModelOpt 的 `FakeTensorQuantFunction` 有 ONNX symbolic，将 fake quant 转成 `QuantizeLinear` → `DequantizeLinear`。[`validate_and_summarize_onnx()` 第 182 行起](../../tools/quantize_vit_modelopt.py#L182) 执行 `onnx.checker.check_model()` 并检查 Q/DQ 节点及权重/激活输入连线。

## 从真实 MLP Linear 对照 Quantizer 与 ONNX

以 `blocks.0.mlp.fc1` 为例，运行结果见 `quant_summary.txt` 第 164、166 行附近：

- `input_quantizer`: 8 bit、`axis=None`、`amax=41.954811`，因此 activation scale 为 `41.954811 / 127 = 0.33035284`。
- `weight_quantizer`: 8 bit、`axis=0`、amax shape `(3072,1)`；weight scale 逐输出通道计算，shape `(3072,1)`，范围约 `1.3561e-9` 到 `2.5672e-3`。

`onnx_summary.txt` 记录到对应图路径：

```text
activation:
.../fc1/input_quantizer/QuantizeLinear
→ .../fc1/input_quantizer/DequantizeLinear
→ MatMul input 0

weight:
blocks.0.mlp.fc1.weight
→ .../fc1/weight_quantizer/QuantizeLinear (axis=0)
→ .../fc1/weight_quantizer/DequantizeLinear
→ Transpose
→ MatMul input 1
```

这说明 Linear 的权重和输入在 ONNX 中都显式经过量化边界。Q 节点输出为 INT8，DQ 输出恢复为 FLOAT 供后续图连接，因此图的最终输入/输出标 FLOAT 并不表示中间没有 INT8 表示。最终 TensorRT 是否把 Q/DQ 融合成 INT8 kernel，仍需受支持版本成功解析并查看 engine；当前没有 8.6.4 Parser/engine 结果，不能据此宣称 TensorRT 的实际执行精度。

## 本次导出统计

- ONNX opset：17；IR version：8。
- 图节点：952；`MatMul` 72 个，`Gemm` 1 个，`Conv` 1 个。
- Q/DQ：100 / 100；其中激活 50 组 per-tensor（`axis=None`），权重 50 组 per-channel（`axis=0`）。
- 所有 scale 为 FLOAT32；zero point 为有符号 INT8 且全为 0。
- 49 个 MatMul/Gemm 的权重输入有可追踪的 initializer→Q→DQ 路径；49 个线性输入直接由 DQ 提供。其余 24 个 attention BMM 仍在图中，但没有量化权重 Q/DQ 路径。
- FP32 与 INT8 ONNX 均通过 ONNX Checker；FP32 图有 0 Q/DQ。

TensorRT 8.5.3 parser 在第 29 个节点 `LayerNormalization` 报错，提示找不到相应 importer/plugin。该日志只代表缓存的 8.5.3 parser 结果，不能代替缺失的 TensorRT 8.6.4 parser 结论。未评价 top-1/top-5 accuracy；现有校准数组没有标签，且来源无法核实。
