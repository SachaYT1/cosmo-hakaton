"""Модель детекции активного горения (AF).

Вход: чип 256x256, in_channels каналов (5 VIIRS I1-I5 + вспомогательные слои,
см. configs/af.yaml -> data.channels + data.aux_channels).
Выход: логиты (B, 1, H, W) — бинарная маска горения (сигмоида применяется в loss/инференсе).

Лёгкий U-Net на 3 уровня: датасет всего ~420 чипов, глубокая сеть переобучится,
а претрейн-энкодеры (ImageNet) бесполезны — входные каналы не RGB
(яркостные температуры, отражение SWIR, метео и т.д.).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Свёртка x2 + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    """U-Net на 3 уровня понижения разрешения, база каналов 32."""

    def __init__(self, in_channels: int, n_classes: int = 1, base: int = 32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.up1 = Up(base * 8, base * 4, base * 4)
        self.up2 = Up(base * 4, base * 2, base * 2)
        self.up3 = Up(base * 2, base, base)

        self.out_conv = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)
