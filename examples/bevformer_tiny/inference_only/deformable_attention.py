"""Pure PyTorch multi-scale deformable attention for inference validation."""

import torch
import torch.nn.functional as F


def multi_scale_deformable_attention(value, spatial_shapes, level_start_index,
                                     sampling_locations, attention_weights,
                                     im2col_step=64):
    """Sample every level and reduce to [batch, queries, heads * channels].

    The six arguments match the original BEVFormer CUDA Function contract.
    Coordinates are normalized to [0, 1], ordered as (x, y).
    """
    if value.ndim != 4 or spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
        raise ValueError('value and spatial_shapes must be [B,S,H,D] and [L,2]')
    if sampling_locations.ndim != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError('sampling_locations must be [B,Q,H,L,P,2]')
    if attention_weights.shape != sampling_locations.shape[:-1]:
        raise ValueError('attention_weights must be [B,Q,H,L,P]')
    batch, samples, heads, channels = value.shape
    location_batch, queries, location_heads, levels, points, _ = sampling_locations.shape
    if location_batch != batch or location_heads != heads or levels != spatial_shapes.shape[0]:
        raise ValueError('batch, heads, or levels differ between inputs')
    if level_start_index.ndim != 1 or level_start_index.shape[0] != levels:
        raise ValueError('level_start_index must be [L]')
    if im2col_step <= 0:
        raise ValueError('im2col_step must be positive')
    if not value.is_floating_point() or sampling_locations.dtype != value.dtype or attention_weights.dtype != value.dtype:
        raise ValueError('value, locations, and weights must share a floating dtype')

    sizes = [(int(height), int(width)) for height, width in spatial_shapes.tolist()]
    counts = [height * width for height, width in sizes]
    starts = [int(index) for index in level_start_index.tolist()]
    expected_starts = [0]
    for count in counts[:-1]:
        expected_starts.append(expected_starts[-1] + count)
    if any(height <= 0 or width <= 0 for height, width in sizes):
        raise ValueError('spatial dimensions must be positive')
    if sum(counts) != samples or starts != expected_starts:
        raise ValueError('value length or level_start_index does not match spatial_shapes')

    sampling_grids = sampling_locations * 2 - 1
    sampled_levels = []
    for level, ((height, width), level_value) in enumerate(zip(sizes, value.split(counts, dim=1))):
        feature_map = level_value.flatten(2).transpose(1, 2).reshape(
            batch * heads, channels, height, width)
        grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_levels.append(F.grid_sample(
            feature_map, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False))

    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points)
    sampled = torch.stack(sampled_levels, dim=-2).flatten(-2)
    output = (sampled * weights).sum(-1).reshape(batch, heads * channels, queries)
    return output.transpose(1, 2).contiguous()
