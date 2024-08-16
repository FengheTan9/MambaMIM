import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn3d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose3d(cin, cin, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True), bn3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=True), bn3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class FusionBlock(nn.Module):
    def __init__(self, cin, cout, bn3d):
        """
        a fusionBlock block with 2x up sampling
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True), bn3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=True), bn3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn3d = nn.BatchNorm3d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn3d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.fuse = nn.ModuleList([FusionBlock(cin * 2, cin, bn3d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                if isinstance(x, int):
                    x = x + to_dec[i]
                else:
                    x = torch.cat((x, to_dec[i]), dim=1)
                    x = self.fuse[i](x)
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
