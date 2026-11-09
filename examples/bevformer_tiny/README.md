# BEVFormer-tiny independent FP32 inference

This example contains the checkpoint-compatible, pure PyTorch BEVFormer-tiny
inference model copied from `BEVFormer/tools/inference_only` at commit `c6348b2`.
It does not import OpenMMLab or enable ModelOpt quantization.

From the Model-Optimizer repository root, run with the `modelopt041` Python:

```bash
/data/conda_envs/modelopt041/bin/python -m examples.bevformer_tiny.run_fp32 \
  --checkpoint /data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth \
  --reference ../BEVFormer/golden_tensors_fp32/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3
```

The runner loads all 643 checkpoint entries with `strict=True`, disables TF32,
and passes its own `bev_embed` to the next frame. It writes per-frame
MaxAE/MAE/MSE for C5, FPN, BEV, raw detection outputs, and decoded detections
to `outputs/bevformer_tiny_fp32_alignment.json`.

The checkpoint and reference tensors are external inputs. The original
BEVFormer source, Golden environment, and historical Golden files remain in
the sibling `BEVFormer` repository.
