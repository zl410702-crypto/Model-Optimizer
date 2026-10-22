# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File handling and FP16 weight conversion for the DLA surgery pipeline."""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ...logging_config import logger

__all__ = ["convert_initializers_to_fp16", "save_onnx_model"]


def save_onnx_model(
    model: onnx.ModelProto,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    size_threshold: int = 1024,
    verbose: bool = True,
) -> None:
    """Save a loaded DLA model, optionally storing large weights in a sidecar.

    Args:
        model: Loaded model; externalizing weights updates its tensor storage metadata.
        output_path: Destination ONNX file.
        use_external_data: Store large tensor payloads in an external file.
        external_data_name: Relative sidecar path; defaults to the ONNX filename plus ``_data``.
        size_threshold: Minimum tensor size in bytes to externalize.
        verbose: Log the destination.
    """
    if verbose:
        logger.info("Saving DLA model to: %s", output_path)
    output_dir = os.path.dirname(output_path)
    location = Path(external_data_name or os.path.basename(output_path) + "_data")
    if use_external_data:
        if (
            external_data_name == ""
            or location.anchor
            or not location.name
            or ".." in location.parts
            or location == Path(Path(output_path).name)
        ):
            raise ValueError(
                "External data must have a distinct relative path inside the output directory"
            )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not use_external_data:
        onnx.save(model, output_path)
        return
    with tempfile.TemporaryDirectory(dir=output_dir or ".") as temporary:
        staged_model = Path(temporary) / Path(output_path).name
        staged_weights = Path(temporary) / location
        staged_weights.parent.mkdir(parents=True, exist_ok=True)
        _save_external_weights(
            model, staged_model, staged_weights, location.as_posix(), size_threshold
        )
        if staged_weights.exists():
            destination = Path(output_dir or ".") / location
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_weights, destination)
        os.replace(staged_model, output_path)


# TODO: Once ONNX >= 1.22.0 is required, use its external writer instead (onnx/onnx#7790).
def _save_external_weights(model, model_path, weights_path, location, size_threshold):
    onnx.external_data_helper.convert_model_to_external_data(
        model, location=location, size_threshold=size_threshold
    )
    tensors = [
        tensor
        for tensor in onnx.external_data_helper._get_all_tensors(model)
        if onnx.external_data_helper.uses_external_data(tensor) and tensor.HasField("raw_data")
    ]
    if tensors:
        # ONNX 1.21 reopens sidecars in Windows text mode, truncating trailing 0x1A.
        with weights_path.open("wb") as stream:
            for tensor in tensors:
                data, offset = tensor.raw_data, stream.tell()
                stream.write(data)
                onnx.external_data_helper.set_external_data(
                    tensor, location, offset=offset, length=len(data)
                )
                tensor.ClearField("raw_data")
    onnx.save(model, model_path)


def convert_initializers_to_fp16(
    graph: onnx.GraphProto,
    *,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> int:
    """Convert DLA FP32 weights in place, preserving order and FP32 schema parameters.

    Match C++ FP16 rounding and signed canonical NaNs. Validate loaded raw or typed
    payloads before mutation; ``exclude`` names schema parameters that must stay FP32.
    """
    candidates = [
        init
        for init in graph.initializer
        if init.data_type == TensorProto.FLOAT and init.name not in exclude
    ]
    for init in candidates:
        raw = init.HasField("raw_data")
        size = len(init.raw_data) if raw else len(init.float_data)
        expected_size = math.prod(init.dims) * (4 if raw else 1)
        if any(dim < 0 for dim in init.dims) or size != expected_size:
            raise ValueError(f"Invalid FP32 initializer payload for {init.name!r}")
    for init in candidates:
        # Rewritten tensors may retain stale external metadata beside a valid raw payload.
        array = (
            np.frombuffer(init.raw_data, dtype=np.float32).reshape(tuple(init.dims))
            if init.HasField("raw_data")
            else numpy_helper.to_array(init)
        )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = array.astype(np.float16)
        source, target = array.reshape(-1), converted.reshape(-1).view(np.uint16)
        # Bound NaN masks and indexed temporaries for large weight tensors.
        for start in range(0, source.size, 1 << 20):
            chunk = source[start : start + (1 << 20)]
            nan = np.isnan(chunk)
            if np.any(nan):
                target[start : start + chunk.size][nan] = (
                    (chunk.view(np.uint32)[nan] >> 16) & 0x8000
                ) | 0x7E00
        init.CopyFrom(numpy_helper.from_array(converted, name=init.name))
    return len(candidates)
