"""BEVFormer-tiny image backbone and neck with checkpoint-compatible names."""

import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, images):
        residual = images
        output = self.relu(self.bn1(self.conv1(images)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            residual = self.downsample(images)
        return self.relu(output + residual)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        in_channels = 64
        for stage, (channels, blocks, stride) in enumerate(
            ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)), 1
        ):
            downsample = None
            if stride != 1 or in_channels != channels * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, channels * Bottleneck.expansion, 1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(channels * Bottleneck.expansion),
                )
            layers = [Bottleneck(in_channels, channels, stride, downsample)]
            layers.extend(Bottleneck(channels * Bottleneck.expansion, channels)
                          for _ in range(1, blocks))
            setattr(self, f'layer{stage}', nn.Sequential(*layers))
            in_channels = channels * Bottleneck.expansion

    def forward(self, images):
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            features = stage(features)
        return features,


class ConvOnly(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding)

    def forward(self, features):
        return self.conv(features)


class SingleLevelFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvOnly(2048, 256, 1)])
        self.fpn_convs = nn.ModuleList([ConvOnly(256, 256, 3, padding=1)])

    def forward(self, features):
        lateral = self.lateral_convs[0](features[0])
        return self.fpn_convs[0](lateral),


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet50()
        self.img_neck = SingleLevelFPN()

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 6:
            raise ValueError('images must have shape [batch, 6, 3, height, width]')
        batch, cameras, channels, height, width = images.shape
        features = self.img_neck(self.img_backbone(
            images.reshape(batch * cameras, channels, height, width)))[0]
        return features.reshape(batch, cameras, *features.shape[1:])
