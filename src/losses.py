from __future__ import annotations

import torch.nn as nn


def create_criterion(loss_name: str) -> nn.Module:
    name = loss_name.strip().upper()

    if name == "L1":
        return nn.L1Loss()

    if name in {"L2", "MSE"}:
        return nn.MSELoss()

    raise ValueError(f"Nežinoma loss funkcija: {loss_name}")
