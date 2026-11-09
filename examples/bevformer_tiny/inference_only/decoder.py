"""Inference-only six-layer BEVFormer decoder with original state_dict names."""

import torch
from torch import nn

from .decoder_cross_attention import CustomMSDeformableAttention


def inverse_sigmoid(values, eps=1e-5):
    clipped = values.clamp(min=0, max=1)
    return torch.log(clipped.clamp(min=eps) / (1 - clipped).clamp(min=eps))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None, attn_mask=None,
                key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        attended = self.attn(query, key, value, attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask)[0]
        return identity + self.dropout_layer(self.proj_drop(attended))


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=512, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels),
                          nn.ReLU(inplace=True), nn.Dropout(dropout)),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout))

    def forward(self, query):
        return query + self.layers(query)


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([
            MultiheadAttention(), CustomMSDeformableAttention()])
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])

    def forward(self, query, key=None, value=None, query_pos=None,
                key_pos=None, attn_masks=None,
                query_key_padding_mask=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        del kwargs
        if attn_masks is None:
            attn_masks = (None, None)
        if len(attn_masks) != 2:
            raise ValueError('attn_masks must contain self and cross attention masks')
        query = self.attentions[0](
            query, query, query, query_pos=query_pos, key_pos=query_pos,
            attn_mask=attn_masks[0], key_padding_mask=query_key_padding_mask)
        query = self.norms[0](query)
        query = self.attentions[1](
            query, key, value, query_pos=query_pos, key_padding_mask=key_padding_mask,
            reference_points=reference_points, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norms[1](query)
        query = self.ffns[0](query)
        return self.norms[2](query)


class DetectionTransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList(
            DetrTransformerDecoderLayer() for _ in range(num_layers))
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None,
                reference_points=None, reg_branches=None, key_padding_mask=None,
                **kwargs):
        if reference_points is None:
            raise ValueError('reference_points are required')
        output = query
        intermediate = []
        intermediate_references = []
        for layer_index, layer in enumerate(self.layers):
            reference_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output, key, value, query_pos=query_pos, key_pos=key_pos,
                reference_points=reference_input,
                key_padding_mask=key_padding_mask, **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                regression = reg_branches[layer_index](output)
                if reference_points.shape[-1] != 3:
                    raise ValueError('regression refinement requires 3D reference points')
                refined = torch.zeros_like(reference_points)
                refined[..., :2] = regression[..., :2] + inverse_sigmoid(reference_points[..., :2])
                refined[..., 2:3] = regression[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = refined.sigmoid().detach()
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_references.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_references)
        return output, reference_points
