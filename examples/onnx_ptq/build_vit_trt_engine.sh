#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TENSORRT_INCLUDE_DIR:-}" || -z "${TENSORRT_LIB_DIR:-}" ]]; then
    echo "Set TENSORRT_INCLUDE_DIR and TENSORRT_LIB_DIR to your TensorRT installation." >&2
    exit 2
fi
cuda_include_dir="${CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
if [[ ! -f "${cuda_include_dir}/cuda_runtime_api.h" ]]; then
    echo "CUDA headers not found under ${cuda_include_dir}; set CUDA_INCLUDE_DIR." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="${BUILD_OUTPUT:-${script_dir}/build_vit_trt_engine}"

c++ -std=c++14 -O2 -Wall -Wextra \
    -I"${TENSORRT_INCLUDE_DIR}" \
    -I"${cuda_include_dir}" \
    "${script_dir}/build_vit_trt_engine.cpp" \
    -L"${TENSORRT_LIB_DIR}" \
    -Wl,-rpath,"${TENSORRT_LIB_DIR}" \
    -lnvonnxparser -lnvinfer \
    -o "${output}"

echo "Built ${output}"
