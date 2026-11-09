"""Standalone BEVFormer-tiny transformer with explicit runtime geometry."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import DetectionTransformerDecoder
from .encoder import BEVFormerEncoder


def rotate_bev_nearest(features, angle_degrees, center=(100, 100)):
    if features.ndim != 3 or angle_degrees.numel() != 1:
        raise ValueError('rotation expects [channels, height, width] and one angle')
    _, height, width = features.shape
    angle = angle_degrees.reshape(()) * (math.pi / 180.0)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    center_x = float(center[0]) - width * 0.5
    center_y = float(center[1]) - height * 0.5
    translate_x = center_x - cosine * center_x + sine * center_y
    translate_y = center_y - sine * center_x - cosine * center_y
    theta = torch.stack((cosine, -sine, translate_x,
                         sine, cosine, translate_y)).reshape(1, 2, 3)
    x_grid = torch.arange(width, device=features.device, dtype=features.dtype) - width * 0.5 + 0.5
    y_grid = torch.arange(height, device=features.device, dtype=features.dtype) - height * 0.5 + 0.5
    grid_y, grid_x = torch.meshgrid(y_grid, x_grid)
    base_grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), -1)
    scale = features.new_tensor([0.5 * width, 0.5 * height])
    grid = base_grid.reshape(1, height * width, 3).bmm(theta.transpose(1, 2) / scale)
    grid = grid.reshape(1, height, width, 2)
    return F.grid_sample(features.unsqueeze(0), grid, mode='nearest',
                         padding_mode='zeros', align_corners=False).squeeze(0)


def compute_shift(can_bus, grid_length, bev_h, bev_w):
    motion = can_bus.to(torch.float64)
    delta_x, delta_y = motion[:, 0], motion[:, 1]
    ego_angle = motion[:, -2] / math.pi * 180.0
    translation_length = torch.sqrt(delta_x.square() + delta_y.square())
    translation_angle = torch.atan2(delta_y, delta_x) / math.pi * 180.0
    angle = (ego_angle - translation_angle) / 180.0 * math.pi
    shift_y = translation_length * torch.cos(angle) / grid_length[0] / bev_h
    shift_x = translation_length * torch.sin(angle) / grid_length[1] / bev_w
    return torch.stack((shift_x, shift_y), -1).to(can_bus.dtype)


class PerceptionTransformer(nn.Module):
    def __init__(self, embed_dims=256, num_feature_levels=4, num_cams=6,
                 rotate_prev_bev=True, use_shift=True, use_can_bus=True,
                 use_cams_embeds=True, rotate_center=(100, 100)):
        super().__init__()
        self.encoder = BEVFormerEncoder()
        self.decoder = DetectionTransformerDecoder()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = tuple(rotate_center)
        self.level_embeds = nn.Parameter(torch.empty(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.empty(num_cams, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims), nn.ReLU(inplace=True))
        self.can_bus_mlp.add_module('norm', nn.LayerNorm(embed_dims))

    def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w,
                         bev_pos, can_bus, lidar2img, image_shape,
                         grid_length=(0.512, 0.512), prev_bev=None, shift=None):
        batch = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, batch, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        if shift is None:
            shift = compute_shift(can_bus, grid_length, bev_h, bev_w)
        if not self.use_shift:
            shift = torch.zeros_like(shift)
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = prev_bev.clone()
            if self.rotate_prev_bev:
                for index in range(batch):
                    history = prev_bev[:, index].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    history = rotate_bev_nearest(history, can_bus[index, -1:], self.rotate_center)
                    prev_bev[:, index] = history.permute(1, 2, 0).reshape(bev_h * bev_w, -1)
        embedding = self.can_bus_mlp(can_bus.to(bev_queries.dtype))[None]
        bev_queries = bev_queries + embedding * self.use_can_bus

        flattened = []
        shapes = []
        for level, features in enumerate(mlvl_feats):
            _, cameras, _, height, width = features.shape
            if cameras != self.num_cams:
                raise ValueError('camera count does not match transformer configuration')
            features = features.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                features = features + self.cams_embeds[:, None, None, :].to(features.dtype)
            features = features + self.level_embeds[None, None, level:level + 1, :].to(features.dtype)
            flattened.append(features)
            shapes.append((height, width))
        spatial_shapes = torch.tensor(shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(1),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))
        features = torch.cat(flattened, 2).permute(0, 2, 1, 3)
        return self.encoder(
            bev_queries, features, features, bev_h, bev_w, bev_pos,
            spatial_shapes, level_start_index, lidar2img, image_shape,
            prev_bev=prev_bev, shift=shift)

    def forward(self, mlvl_feats, bev_queries, object_query_embed, bev_h, bev_w,
                bev_pos, can_bus, lidar2img, image_shape, reg_branches=None,
                grid_length=(0.512, 0.512), prev_bev=None, shift=None):
        bev_embed = self.get_bev_features(
            mlvl_feats, bev_queries, bev_h, bev_w, bev_pos, can_bus,
            lidar2img, image_shape, grid_length, prev_bev, shift)
        batch = mlvl_feats[0].shape[0]
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch, -1, -1)
        query = query.unsqueeze(0).expand(batch, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        states, refinements = self.decoder(
            query=query, value=bev_embed, query_pos=query_pos,
            reference_points=reference_points, reg_branches=reg_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device))
        return bev_embed, states, reference_points, refinements
