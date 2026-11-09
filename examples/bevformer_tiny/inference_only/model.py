"""Standalone BEVFormer-tiny forward model with checkpoint-compatible names."""

import copy

import torch
from torch import nn

from .decoder import inverse_sigmoid
from .image_encoder import ResNet50, SingleLevelFPN
from .transformer import PerceptionTransformer


PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats=128, row_num_embed=50, col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)

    def forward(self, mask):
        batch, height, width = mask.shape
        x_positions = torch.arange(width, device=mask.device)
        y_positions = torch.arange(height, device=mask.device)
        x_embeddings = self.col_embed(x_positions)
        y_embeddings = self.row_embed(y_positions)
        position = torch.cat((
            x_embeddings.unsqueeze(0).repeat(height, 1, 1),
            y_embeddings.unsqueeze(1).repeat(1, width, 1)), -1)
        return position.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)


class BEVFormerHead(nn.Module):
    def __init__(self, bev_h=50, bev_w=50, num_query=900,
                 num_classes=10, code_size=10, pc_range=PC_RANGE):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = tuple(pc_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.transformer = PerceptionTransformer()
        self.positional_encoding = LearnedPositionalEncoding(
            row_num_embed=bev_h, col_num_embed=bev_w)
        classification = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))
        regression = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, code_size))
        self.cls_branches = nn.ModuleList(
            copy.deepcopy(classification) for _ in range(6))
        self.reg_branches = nn.ModuleList(
            copy.deepcopy(regression) for _ in range(6))
        self.bev_embedding = nn.Embedding(bev_h * bev_w, 256)
        self.query_embedding = nn.Embedding(num_query, 512)
        self.code_weights = nn.Parameter(torch.tensor(
            [1.0] * 8 + [0.2, 0.2]), requires_grad=False)

    def forward(self, mlvl_feats, can_bus, lidar2img, image_shape,
                prev_bev=None, shift=None):
        batch = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        object_queries = self.query_embedding.weight.to(dtype)
        mask = bev_queries.new_zeros(batch, self.bev_h, self.bev_w)
        bev_pos = self.positional_encoding(mask).to(dtype)
        bev_embed, states, initial_references, refinements = self.transformer(
            mlvl_feats, bev_queries, object_queries, self.bev_h, self.bev_w,
            bev_pos, can_bus, lidar2img, image_shape,
            reg_branches=self.reg_branches,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            prev_bev=prev_bev, shift=shift)
        states = states.permute(0, 2, 1, 3)
        classes = []
        boxes = []
        for level in range(states.shape[0]):
            reference = initial_references if level == 0 else refinements[level - 1]
            reference = inverse_sigmoid(reference)
            classes.append(self.cls_branches[level](states[level]))
            coordinates = self.reg_branches[level](states[level])
            coordinates[..., :2] += reference[..., :2]
            coordinates[..., :2] = coordinates[..., :2].sigmoid()
            coordinates[..., 4:5] += reference[..., 2:3]
            coordinates[..., 4:5] = coordinates[..., 4:5].sigmoid()
            coordinates[..., 0:1] = coordinates[..., 0:1] * self.real_w + self.pc_range[0]
            coordinates[..., 1:2] = coordinates[..., 1:2] * self.real_h + self.pc_range[1]
            coordinates[..., 4:5] = coordinates[..., 4:5] * (
                self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            boxes.append(coordinates)
        return {'bev_embed': bev_embed, 'all_cls_scores': torch.stack(classes),
                'all_bbox_preds': torch.stack(boxes)}


class BEVFormerTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet50()
        self.img_neck = SingleLevelFPN()
        self.pts_bbox_head = BEVFormerHead()

    def extract_img_feat(self, images):
        if images.ndim != 5 or images.shape[1] != 6:
            raise ValueError('images must be [batch, 6, 3, height, width]')
        batch, cameras, channels, height, width = images.shape
        features = images.reshape(batch * cameras, channels, height, width)
        features = self.img_neck(self.img_backbone(features))[0]
        return (features.reshape(batch, cameras, *features.shape[1:]),)

    def forward(self, images, can_bus, lidar2img, image_shape,
                prev_bev=None, shift=None):
        features = self.extract_img_feat(images)
        return self.pts_bbox_head(features, can_bus, lidar2img, image_shape,
                                  prev_bev=prev_bev, shift=shift)
