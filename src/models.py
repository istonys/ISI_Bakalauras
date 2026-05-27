from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.net(x))
        linear_mag = torch.expm1(x)
        masked_mag = linear_mag * mask
        return torch.log1p(masked_mag)


class DilatedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.net(x))
        linear_mag = torch.expm1(x)
        masked_mag = linear_mag * mask
        return torch.log1p(masked_mag)


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        z = self.encoder(x)
        y = self.decoder(z)
        if y.shape[-2:] != original_size:
            y = F.interpolate(y, size=original_size, mode="bilinear", align_corners=False)
        return y


class UNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())

        self.out = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        y = self.out(d1)
        if y.shape[-2:] != original_size:
            y = F.interpolate(y, size=original_size, mode="bilinear", align_corners=False)
        return y


class UNetMaskCNN(nn.Module):
    """UNet architecture that predicts a soft mask (like MaskCNN) instead of direct magnitude.
    The mask is applied in the linear magnitude domain, then re-log-compressed."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())

        # Output: sigmoid mask [0,1]
        self.mask_out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        if d1.shape[-2:] != original_size:
            d1 = F.interpolate(d1, size=original_size, mode="bilinear", align_corners=False)

        # Mask prediction: sigmoid -> [0,1]
        mask = torch.sigmoid(self.mask_out(d1))
        # Apply mask in linear magnitude domain
        linear_mag = torch.expm1(x)
        masked_mag = linear_mag * mask
        return torch.log1p(masked_mag)



class UNetDilatedMaskCNN(nn.Module):
    """UNet su dilated konvoliucijomis bottleneck'e ir sigmoid maske.
    Jungia UNetMaskCNN (skip connections + mask) ir DilatedMaskCNN (platiau receptyvus laukas)."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        # Dilated bottleneck: receptyvus laukas 1->2->4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, dilation=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())

        self.mask_out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        if d1.shape[-2:] != original_size:
            d1 = F.interpolate(d1, size=original_size, mode="bilinear", align_corners=False)

        mask = torch.sigmoid(self.mask_out(d1))
        linear_mag = torch.expm1(x)
        return torch.log1p(linear_mag * mask)

# Registry mapping model_name -> callable that takes n_freq and returns the model.
# All entries take n_freq for a uniform calling convention, even if they ignore it.
_MODEL_REGISTRY = {
    "BaseCNN": lambda n_freq: BaseCNN(),
    "MaskCNN": lambda n_freq: MaskCNN(),
    "DilatedCNN": lambda n_freq: DilatedCNN(),
    "DilatedMaskCNN": lambda n_freq: DilatedMaskCNN(),
    "CNNAutoencoder": lambda n_freq: CNNAutoencoder(),
    "UNetCNN": lambda n_freq: UNetCNN(),
    "UNetMaskCNN": lambda n_freq: UNetMaskCNN(),
    "UNetDilatedMaskCNN": lambda n_freq: UNetDilatedMaskCNN(),
}


def create_model(model_name: str, n_freq: int) -> nn.Module:
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Nezinomas modelis: {model_name}. "
            f"Galimi: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_name](n_freq)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
