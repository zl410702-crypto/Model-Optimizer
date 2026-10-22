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

"""DLA model serialization and C++ FP16 weight policy."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.utils.dla_utils import (
    convert_initializers_to_fp16,
    save_onnx_model,
)


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
def test_repeated_external_save_keeps_sidecar_bounded(tmp_path, dtype):
    """Round-trip adjacent payloads, including a Windows text-mode EOF byte."""
    path = tmp_path / "model.onnx"
    location = "weights/data.bin" if dtype == np.float32 else "model.onnx_data"
    values = np.arange(2048, dtype=np.float32).astype(dtype)
    for offset in [0, 1]:
        first, second = values + offset, np.full_like(values, 39 + offset)
        first[-1] = 26
        model = helper.make_model(
            helper.make_graph(
                [],
                "weights",
                [],
                [
                    helper.make_tensor_value_info(
                        name, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), [2048]
                    )
                    for name in ("w", "other")
                ],
                [numpy_helper.from_array(first, "w"), numpy_helper.from_array(second, "other")],
            )
        )
        save_onnx_model(model, str(path), external_data_name=location, verbose=False)
        assert (tmp_path / location).stat().st_size == 2 * values.nbytes
        saved = onnx.load(path)
        for tensor, expected in zip(saved.graph.initializer, (first, second), strict=True):
            np.testing.assert_array_equal(numpy_helper.to_array(tensor), expected)


@pytest.mark.parametrize("location", ["../weights.bin", "absolute", "model.onnx", ""])
def test_invalid_external_location_preserves_model_and_existing_output(tmp_path, location):
    model = helper.make_model(
        helper.make_graph(
            [], "save", [], [], [numpy_helper.from_array(np.ones(2048, np.float32), "w")]
        )
    )
    before = model.SerializeToString()
    path = tmp_path / "model.onnx"
    path.write_bytes(b"unchanged")
    if location == "absolute":
        location = str(tmp_path / "weights.bin")
    with pytest.raises(ValueError):
        save_onnx_model(model, str(path), external_data_name=location, verbose=False)
    assert model.SerializeToString() == before
    assert path.read_bytes() == b"unchanged"


@pytest.mark.parametrize("raw", [False, True])
def test_fp16_conversion_preserves_order_exclusions_and_precision(raw):
    values = np.array(
        [0.25, 0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 65520, 2**-24, 2**-25], np.float32
    )
    weight = (
        numpy_helper.from_array(values, "w")
        if raw
        else helper.make_tensor("w", TensorProto.FLOAT, values.shape, values)
    )
    graph = helper.make_graph(
        [],
        "conversion",
        [],
        [],
        [
            weight,
            numpy_helper.from_array(np.array([1], np.int64), "axes"),
            numpy_helper.from_array(np.array([1], np.float32), "scales"),
        ],
    )
    original = graph.initializer[0]
    untouched = [t.SerializeToString() for t in graph.initializer[1:]]
    assert convert_initializers_to_fp16(graph, exclude={"scales"}) == 1
    assert [t.name for t in graph.initializer] == ["w", "axes", "scales"]
    assert graph.initializer[0] is original
    assert original.data_type == TensorProto.FLOAT16
    assert [t.SerializeToString() for t in graph.initializer[1:]] == untouched
    expected = [0x3400, 0, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C00, 1, 0]
    np.testing.assert_array_equal(np.frombuffer(original.raw_data, dtype=np.uint16), expected)


@pytest.mark.parametrize("raw", [False, True])
def test_invalid_fp16_conversion_refuses_before_mutation(raw):
    good = numpy_helper.from_array(np.array([1], np.float32), "good")
    payload = {"raw_data": b"1234"} if raw else {"float_data": [1]}
    invalid = TensorProto(name="invalid", data_type=TensorProto.FLOAT, dims=[2], **payload)
    graph = helper.make_graph([], "invalid", [], [], [good, invalid])
    before = graph.SerializeToString()
    with pytest.raises(ValueError):
        convert_initializers_to_fp16(graph)
    assert graph.SerializeToString() == before
