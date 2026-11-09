"""Checkpoint-compatible spatial cross-attention without OpenMMLab."""

import torch
from torch import nn

from .deformable_attention import multi_scale_deformable_attention


class MSDeformableAttention3D(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1,
                 num_points=8, batch_first=True, im2col_step=64):
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None, **kwargs):
        del key, identity, kwargs
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        batch, queries, channels = query.shape
        num_value = value.shape[1]
        if channels != self.embed_dims or reference_points is None:
            raise ValueError('invalid query channels or missing reference_points')
        if reference_points.shape[:2] != (batch, queries) or reference_points.shape[-1] != 2:
            raise ValueError('reference_points must be [batch, queries, anchors, 2]')
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

        anchors = reference_points.shape[2]
        if self.num_points % anchors:
            raise ValueError('num_points must be divisible by reference anchors')
        normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
        offsets = (offsets / normalizer[None, None, None, :, None, :]).reshape(
            batch, queries, self.num_heads, self.num_levels,
            self.num_points // anchors, anchors, 2)
        locations = (reference_points[:, :, None, None, None, :, :] + offsets).reshape(
            batch, queries, self.num_heads, self.num_levels, self.num_points, 2)
        output = multi_scale_deformable_attention(
            projected_value, spatial_shapes, level_start_index, locations,
            weights, self.im2col_step)
        return output if self.batch_first else output.permute(1, 0, 2)


class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims=256, num_cams=6, dropout=0.1,
                 num_heads=8, num_levels=1, num_points=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims, num_heads, num_levels, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key, value, residual=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                reference_points_cam=None, bev_mask=None,
                level_start_index=None, **kwargs):
        del key_padding_mask, reference_points, kwargs
        if key is None:
            key = query
        if value is None:
            value = key
        inp_residual = query if residual is None else residual
        if query_pos is not None:
            query = query + query_pos
        batch, queries, channels = query.shape
        if channels != self.embed_dims or reference_points_cam is None or bev_mask is None:
            raise ValueError('invalid query channels or missing camera references/mask')
        if reference_points_cam.shape[:3] != (self.num_cams, batch, queries):
            raise ValueError('reference_points_cam must be [cameras, batch, queries, anchors, 2]')
        if bev_mask.shape[:3] != (self.num_cams, batch, queries):
            raise ValueError('bev_mask must be [cameras, batch, queries, anchors]')

        indexes = [mask[0].sum(-1).nonzero().squeeze(-1) for mask in bev_mask]
        max_length = max(len(indices) for indices in indexes)
        if max_length == 0:
            raise ValueError('no camera sees any BEV query')
        anchors = reference_points_cam.shape[3]
        rebatch_query = query.new_zeros(batch, self.num_cams, max_length, channels)
        rebatch_reference = reference_points_cam.new_zeros(
            batch, self.num_cams, max_length, anchors, 2)
        for batch_index in range(batch):
            for camera_index, indices in enumerate(indexes):
                length = len(indices)
                rebatch_query[batch_index, camera_index, :length] = query[batch_index, indices]
                rebatch_reference[batch_index, camera_index, :length] = (
                    reference_points_cam[camera_index, batch_index, indices])

        cameras, num_value, key_batch, _ = key.shape
        if cameras != self.num_cams or key_batch != batch or value.shape != key.shape:
            raise ValueError('key/value must be [cameras, samples, batch, channels]')
        key = key.permute(2, 0, 1, 3).reshape(batch * cameras, num_value, channels)
        value = value.permute(2, 0, 1, 3).reshape(batch * cameras, num_value, channels)
        sampled = self.deformable_attention(
            rebatch_query.reshape(batch * cameras, max_length, channels),
            key=key, value=value,
            reference_points=rebatch_reference.reshape(batch * cameras, max_length, anchors, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).reshape(batch, cameras, max_length, channels)

        slots = torch.zeros_like(query)
        for batch_index in range(batch):
            for camera_index, indices in enumerate(indexes):
                slots[batch_index, indices] += sampled[batch_index, camera_index, :len(indices)]
        count = (bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1).clamp(min=1.0)
        slots = self.output_proj(slots / count[..., None])
        return self.dropout(slots) + inp_residual
