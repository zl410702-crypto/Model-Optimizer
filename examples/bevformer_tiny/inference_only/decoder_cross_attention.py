"""Inference-only decoder deformable cross-attention."""

import torch
from torch import nn

from .deformable_attention import multi_scale_deformable_attention


class CustomMSDeformableAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1,
                 num_points=4, dropout=0.1, batch_first=False,
                 im2col_step=64):
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None, **kwargs):
        del key, kwargs
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        batch, queries, channels = query.shape
        num_value = value.shape[1]
        if channels != self.embed_dims or reference_points is None:
            raise ValueError('invalid query channels or missing reference_points')
        if reference_points.ndim != 4 or reference_points.shape[:3] != (
                batch, queries, self.num_levels):
            raise ValueError('reference_points must be [batch, queries, levels, 2/4]')
        if spatial_shapes is None or level_start_index is None:
            raise ValueError('spatial metadata is required')

        projected_value = self.value_proj(value)
        if key_padding_mask is not None:
            projected_value = projected_value.masked_fill(key_padding_mask[..., None], 0.0)
        projected_value = projected_value.reshape(
            batch, num_value, self.num_heads, channels // self.num_heads)
        offsets = self.sampling_offsets(query).reshape(
            batch, queries, self.num_heads, self.num_levels, self.num_points, 2)
        weights = self.attention_weights(query).reshape(
            batch, queries, self.num_heads, self.num_levels * self.num_points)
        weights = weights.softmax(-1).reshape(
            batch, queries, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
            locations = reference_points[:, :, None, :, None, :] + (
                offsets / normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            locations = reference_points[:, :, None, :, None, :2] + (
                offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError('reference_points last dimension must be 2 or 4')

        output = multi_scale_deformable_attention(
            projected_value, spatial_shapes, level_start_index, locations,
            weights, self.im2col_step)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
