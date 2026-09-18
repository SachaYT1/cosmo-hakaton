"""Модель картирования гарей и степени поражения (BS).

Сиамский энкодер: снимки "до" и "после" (Sentinel-2 + Sentinel-1 + NBR
на каждую дату) проходят через ОДИН И ТОТ ЖЕ энкодер (общие веса), чтобы
явно моделировать "было/стало", а не заставлять сеть заново открывать
разность на 224 примерах. На каждом уровне энкодера признаки pre/post
объединяются как [pre, post, |pre-post|] — abs-разность подчёркивает
именно изменение, которое и есть сигнал гари.

Статические слои без даты (рельеф, тип покрова) подмешиваются в bottleneck,
т.к. они не участвуют в сравнении "до/после".

Выход: логиты (B, n_classes, H, W), n_classes=4 (0 не горело, 1/2/3 степени).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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


class Encoder(nn.Module):
    """Общий энкодер на 4 уровня, применяется к pre и post с одними весами."""

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(base, base * 2)
        self.enc2 = ConvBlock(base * 2, base * 4)
        self.enc3 = ConvBlock(base * 4, base * 8)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.inc(x)
        f1 = self.enc1(self.pool(f0))
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        return [f0, f1, f2, f3]  # мелкий -> глубокий уровень


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _fuse(f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
    """[pre, post, |pre-post|] по каналам — тройной вход в декодер на каждом уровне."""
    return torch.cat([f_pre, f_post, torch.abs(f_pre - f_post)], dim=1)


class UNetSiamese(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        static_channels: int = 0,
        n_classes: int = 4,
        base: int = 32,
    ):
        """
        branch_channels — каналов в ОДНОЙ дате (S2 + S1 + NBR этой даты).
        static_channels — каналов без даты (DEM, уклон, экспозиция, тип покрова),
            подмешиваются в bottleneck; поставь 0, если решишь не использовать.
        """
        super().__init__()
        self.encoder = Encoder(branch_channels, base)

        self.static_proj = (
            nn.Conv2d(static_channels, base * 8, kernel_size=1) if static_channels > 0 else None
        )

        # skip-каналы декодера = 3x (pre, post, |diff|) на каждом уровне энкодера
        self.up1 = Up(base * 8 * 3, base * 4 * 3, base * 4)
        self.up2 = Up(base * 4, base * 2 * 3, base * 2)
        self.up3 = Up(base * 2, base * 1 * 3, base)

        self.out_conv = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats_pre = self.encoder(pre)
        feats_post = self.encoder(post)

        skips = [_fuse(fp, fq) for fp, fq in zip(feats_pre[:-1], feats_post[:-1])]
        bottleneck = _fuse(feats_pre[-1], feats_post[-1])

        if self.static_proj is not None and static is not None:
            bottleneck = bottleneck + self.static_proj(static).repeat(1, 3, 1, 1)

        x = self.up1(bottleneck, skips[2])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[0])
        return self.out_conv(x)
