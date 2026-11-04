# ViT ONNX 与 TensorRT INT8 分析

## 结论

目前可以完整分析 INT8 ONNX 中的 Q/DQ 结构，但**不能做 FP16 对 INT8 TensorRT engine 的实测比较**：目录里没有 FP16 ONNX、FP16/INT8 plan/engine、精度验证结果或 benchmark 结果。本机只找到 TensorRT 8.5.3 与 8.4.12 的 Bazel 缓存库；没有 TensorRT 8.6.4、`trtexec` 或可导入的 TensorRT Python 包。尝试用 8.5.3 解析 INT8 ONNX 时，在 `LayerNormalization` 节点失败，engine 没有生成。

因此，本文的 Q/DQ 计数、scale、dtype 和 ONNX 文件尺寸是**实测**；engine tactic、真实计算精度、reformat 数量、engine 显存、准确率和延迟均标为**不可用**，不从 ONNX 的 FLOAT 标注推断 kernel 精度。

## 1. 已有文件与量化流程

| 项目 | 找到的文件 | 已验证内容 |
|---|---|---|
| ViT 基础 ONNX | `examples/onnx_ptq/work_vit/vit_base_patch16_224.onnx` | 模型名对应 ViT-Base/patch16/224；输入 `[1,3,224,224]`，输出 `[1,1000]`；ONNX opset 20；输入、输出及权重为 FLOAT。它是 FP32 ONNX，**不是 FP16 baseline**。 |
| INT8 ONNX | `examples/onnx_ptq/work_vit/vit_base_patch16_224_int8.onnx` | 217 个 `QuantizeLinear`、217 个 `DequantizeLinear`；图输入输出仍标为 FLOAT，内含显式 Q/DQ。 |
| 校准数据 | `examples/onnx_ptq/work_vit/calib64.npy` | `[64,3,224,224]`、float32，数值范围 `[-1,1]`。文件存在不等于能证明这次生成确实使用了它。 |
| FP16 ONNX / TRT engines | 未找到 | 没有可比较的 FP16 ONNX 或任何 ViT `.plan`/`.engine`。 |
| 评估脚本 | `examples/onnx_ptq/evaluate.py` | 可选择 TensorRT 精度、engine 路径、ImageNet 数据路径，并报告 top-1/top-5 与 latency；本机没有相应 TensorRT Python runtime 和既有评估结果。 |

ModelOpt ONNX PTQ 的主流程是：先用校准样本统计激活范围并量化权重，再将 scale 和 zero point 写入 ONNX Q/DQ 边界，最后交给 TensorRT parser/builder 编译成针对 GPU 的 engine。Q/DQ 让 ONNX 显式表达哪些张量要量化；它们本身不是已经编译好的 INT8 kernel。

仓库中的 `modelopt.onnx.quantization.quantize()` 默认 `quantize_mode="int8"`、校准方法 `entropy`、`high_precision_dtype="fp16"`。但此 INT8 ONNX 的图输入、输出和原始权重仍是 FLOAT，producer 是 `onnx.quantize 0.1.0`，目录中也没有生成命令、配置文件或日志。因此**不能把仓库 API 默认值当成该文件的真实生成配置**；图中实际可确认的是 INT8、zero point 为 0 的对称量化。`calib64.npy` 的名字和形状提示它可能是该模型的校准数据，但来源仍未证实。

## 2. INT8 ONNX 中 Q/DQ 的位置

已解析的 INT8 图有 986 个节点，其中 Q/DQ 各 217 个。按 Q 输入是否来自原始权重 initializer 分类：49 组作用于常量权重，168 组作用于激活/中间张量。

| 对象 | 组数 | scale | axis | zero point |
|---|---:|---|---|---|
| MatMul 权重 | 48 | FLOAT 向量；每个输出通道一个 scale，长度按矩阵输出通道为 768、2304 或 3072 | 1；ONNX 权重布局为 `[输入通道,输出通道]` | INT8，全为 0 |
| Patch embedding Conv 权重 | 1 | FLOAT 向量，长度 768 | 0；Conv 权重首维为输出通道 | INT8，全为 0 |
| 激活/中间值 | 168 | FLOAT 标量，每张量一个 scale | 未设置（per-tensor） | INT8 标量，全为 0 |

权重 scale 实测范围约 `1.36e-9` 至 `1.38e-2`；激活 scale 约 `2.78e-3` 至 `1.54`。scale 本身用 FLOAT 保存，zero point 用有符号 INT8 保存。

### Attention 与 MLP

12 个 Transformer block 的 48 个主要线性权重都带 Q/DQ：每个 block 的 `attn.qkv`、`attn.proj`、`mlp.fc1`、`mlp.fc2` 各一个 MatMul 权重。每个 block 的两个 attention batch MatMul（Q×Kᵀ 与 attention×V）也都接收带 Q/DQ 的激活输入。激活边界还出现在 attention 的缩放乘法、残差 Add 等路径上。

`LayerNormalization`、`Softmax`、`Gelu` 周围也能看到量化边界，但这不表示这些节点本身一定以 INT8 执行。例如 LayerNorm 输出之后可以接 Q；TensorRT 可能在可量化的后继 MatMul 前融合 Q/DQ。最终在哪个算子上执行何种 precision，必须检查已构建 engine。

分类头 `/head/Gemm` 的权重和 bias 仍为 FLOAT，没有 weight Q/DQ。LayerNorm 参数、线性层 bias 以及其他未列入 49 组常量权重的参数也保持 FLOAT。

**FLOAT 输出不等于 FLOAT 计算。** Q 的输出是 INT8 量化值，DQ 将其恢复为高精度表示供后续图连接；TensorRT 可识别 Q/DQ 语义，将 Q/DQ 与可量化算子融合。算子累加和输出也可能是更高精度。TensorRT 官方文档明确说明，Q/DQ 网络中的高精度权重可按 Q scale 量化，量化后的权重可保存在 engine plan 中；Q/DQ 也控制量化边界，而非逐节点强制固定某个 kernel。见 [TensorRT Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)。

## 3. TensorRT Engine Inspector

没有可供 Inspector 读取的 ViT engine。尝试用缓存中的 TensorRT 8.5.3 解析 INT8 ONNX，日志显示模型 opset 20，随后报告没有 `LayerNormalization` importer，也没有匹配的 plugin，最终解析失败。因此以下项目都**没有实测结果**：

- engine layer precision 与每层输入/输出 format；
- tactic 选择；
- Q/DQ fusion 结果；
- `Reformat` 层的数量、位置和开销。

TensorRT Inspector 需要一个成功构建并反序列化的 engine；设置 detailed profiling verbosity 后才能查看层参数、precision、format、tactic 等信息。[Engine Tools and Debugging](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html)。这次 parser 在构建阶段之前失败，所以不能从模型图替代 Inspector 结论。

宿主 GPU 实测为 NVIDIA GeForce RTX 4060，8 GiB，driver 580.173.02；主机 GPU 可见。TensorRT 库搜索仅找到 8.5.3 和 8.4.12，未找到优先要求的 8.6.4。8.5.3 的失败是 ONNX parser/operator 支持问题，不是容器 GPU 访问问题；engine 尚未进入 build 阶段。

## 4. FP16 与 INT8 对比

| 指标 | FP16 | INT8 | 结果说明 |
|---|---|---|---|
| Accuracy | 未测 | 未测 | 没有 ImageNet 验证结果。64 张校准输入没有标签，不能计算 top-1/top-5。 |
| Latency | 未测 | 未测 | 没有成功构建的 engine 或可比 benchmark。不能据此判断 INT8 更快或更慢。 |
| Memory usage | 未测 | 未测 | 没有 engine / 推理显存峰值数据。 |
| Actual execution precision | 未知 | 未知 | 没有 Engine Inspector layer report。Q/DQ 只能证明图中指定量化边界，不能证明最终 tactic/kernel。 |
| Q/DQ / Reformat overhead | 未测 | 未测 | 需要 engine 层信息和逐层 profile；ONNX 节点数不能代替融合后层数或 runtime 开销。 |
| ONNX 文件大小 | 没有 FP16 文件 | 346,960,564 bytes（约 330.91 MiB） | 基础 FP32 ONNX 为 346,394,617 bytes（约 330.37 MiB）；INT8 Q/DQ 文件反而大 565,947 bytes（约 0.54 MiB，+0.16%）。这是序列化文件大小，不是 engine 权重或显存大小。 |

INT8 在具备高效 INT8 Tensor Core kernel、且矩阵运算占主导时可能降低延迟与权重带宽。ViT batch=1 时，LayerNorm/Softmax/GELU、attention 中较小的矩阵运算、Q/DQ 边界以及 layout conversion 也可能占据明显比例；额外转换或不利 tactic 会抵消收益。因此只有同 GPU、同输入、同 batch、相同 warmup/iteration 与 timing 条件下的 FP16/INT8 engine 测量才能回答本模型是否更快。TensorRT 也可能把量化权重折入 plan，故 ONNX 文件大小增加并不代表部署 engine 也增加同样多。[TensorRT Performance Measurement](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/measurement-techniques.html)。

## 5. 部署收益判断与下一步

目前**不能证明 INT8 带来有意义的部署收益**：尚无 FP16 baseline engine、INT8 engine、精度结果、延迟结果或显存峰值。已证明的只有 INT8 ONNX 含有 217 组显式 Q/DQ，其中有 49 组 per-channel 权重边界、168 组 per-tensor 激活边界。

继续完成比较至少需要：

1. 在支持 opset 20 `LayerNormalization` 的 TensorRT 版本中成功构建 FP16 与 INT8 engine，并对两者以 detailed verbosity 导出 Inspector JSON；
2. 用同一批带标签图像测 top-1/top-5，用同一组固定输入及相同 batch/计时流程测 latency，并记录推理峰值显存；
3. 从两个 Inspector JSON 中汇总 precision、I/O format、tactic、Q/DQ 融合及 reformat，再与逐层 profile 对照。

本报告没有修改或覆盖任何原始 ONNX、engine 或校准数据。分析输入仅为 `work_vit` 中现有的两个 ONNX 和 `calib64.npy`，以及仓库的量化 API、评估脚本和 ONNX PTQ README。
