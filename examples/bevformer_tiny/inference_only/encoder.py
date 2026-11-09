"""Inference-only BEVFormer encoder with explicit camera geometry inputs."""

import torch
from torch import nn

from .decoder import FFN
from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention


PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


class BEVFormerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([TemporalSelfAttention(), SpatialCrossAttention()])
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])

    def forward(self, query, key, value, bev_pos, ref_2d,
                reference_points_cam, bev_mask, bev_h, bev_w,
                spatial_shapes, level_start_index, prev_bev=None):
        temporal_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        temporal_starts = torch.tensor([0], device=query.device)
        query = self.attentions[0](
            query, prev_bev, prev_bev, query_pos=bev_pos,
            reference_points=ref_2d, spatial_shapes=temporal_shapes,
            level_start_index=temporal_starts)
        query = self.norms[0](query)
        query = self.attentions[1](
            query, key, value, reference_points_cam=reference_points_cam,
            bev_mask=bev_mask, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norms[1](query)
        query = self.ffns[0](query)
        return self.norms[2](query)


class BEVFormerEncoder(nn.Module):
    def __init__(self, num_layers=3, pc_range=PC_RANGE,
                 num_points_in_pillar=4):
        super().__init__()
        self.layers = nn.ModuleList(BEVFormerLayer() for _ in range(num_layers))
        self.num_layers = num_layers
        self.pc_range = tuple(pc_range)
        self.num_points_in_pillar = num_points_in_pillar

    @staticmethod
    def get_reference_points(height, width, z_range=8, num_points_in_pillar=4,
                             dim='3d', batch=1, device='cpu', dtype=torch.float32):
        if dim == '3d':
            z_values = torch.linspace(0.5, z_range - 0.5, num_points_in_pillar,
                                      dtype=dtype, device=device).view(-1, 1, 1)
            z_values = z_values.expand(num_points_in_pillar, height, width) / z_range
            x_values = torch.linspace(0.5, width - 0.5, width,
                                      dtype=dtype, device=device).view(1, 1, width)
            x_values = x_values.expand(num_points_in_pillar, height, width) / width
            y_values = torch.linspace(0.5, height - 0.5, height,
                                      dtype=dtype, device=device).view(1, height, 1)
            y_values = y_values.expand(num_points_in_pillar, height, width) / height
            points = torch.stack((x_values, y_values, z_values), -1)
            points = points.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            return points[None].repeat(batch, 1, 1, 1)
        if dim == '2d':
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device))
            points = torch.stack((grid_x.reshape(-1) / width,
                                  grid_y.reshape(-1) / height), -1)
            return points[None].repeat(batch, 1, 1).unsqueeze(2)
        raise ValueError('dim must be 2d or 3d')

    def point_sampling(self, reference_points, lidar2img, image_shape):
        if lidar2img.ndim != 4 or lidar2img.shape[-2:] != (4, 4):
            raise ValueError('lidar2img must be [batch, cameras, 4, 4]')
        height, width = image_shape
        points = reference_points.clone()
        for axis in range(3):
            points[..., axis] = points[..., axis] * (
                self.pc_range[axis + 3] - self.pc_range[axis]) + self.pc_range[axis]
        points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        points = points.permute(1, 0, 2, 3)
        anchors, batch, queries = points.shape[:3]
        cameras = lidar2img.shape[1]
        points = points.reshape(anchors, batch, 1, queries, 4).repeat(
            1, 1, cameras, 1, 1).unsqueeze(-1)
        matrices = lidar2img.reshape(1, batch, cameras, 1, 4, 4).repeat(
            anchors, 1, 1, queries, 1, 1)
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            projected = torch.matmul(matrices.float(), points.float()).squeeze(-1)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
        depth_valid = projected[..., 2:3] > 1e-5
        xy = projected[..., :2] / projected[..., 2:3].clamp(min=1e-5)
        xy = torch.stack((xy[..., 0] / width, xy[..., 1] / height), -1)
        valid = (depth_valid & (xy[..., 0:1] > 0) & (xy[..., 0:1] < 1)
                 & (xy[..., 1:2] > 0) & (xy[..., 1:2] < 1))
        return xy.permute(2, 1, 3, 0, 4), valid.permute(2, 1, 3, 0, 4).squeeze(-1)

    def forward(self, bev_query, key, value, bev_h, bev_w, bev_pos,
                spatial_shapes, level_start_index, lidar2img, image_shape,
                prev_bev=None, shift=None):
        batch = bev_query.shape[1]
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, '3d', batch, bev_query.device, bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', batch=batch,
            device=bev_query.device, dtype=bev_query.dtype)
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, lidar2img, image_shape)
        if shift is None:
            shift = ref_2d.new_zeros(batch, 2)
        shifted = ref_2d.clone() + shift[:, None, None, :]
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        if prev_bev is not None:
            history = prev_bev.permute(1, 0, 2)
            history = torch.stack((history, bev_query), 1).reshape(batch * 2, bev_h * bev_w, -1)
            hybrid_ref = torch.stack((shifted, ref_2d), 1).reshape(batch * 2, bev_h * bev_w, 1, 2)
        else:
            history = None
            hybrid_ref = torch.stack((ref_2d, ref_2d), 1).reshape(batch * 2, bev_h * bev_w, 1, 2)
        output = bev_query
        for layer in self.layers:
            output = layer(output, key, value, bev_pos, hybrid_ref,
                           reference_points_cam, bev_mask, bev_h, bev_w,
                           spatial_shapes, level_start_index, history)
        return output
