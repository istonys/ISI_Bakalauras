from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any

import os
import yaml


@dataclass
class ExperimentConfig:
    experiment_id: str = "dry_run"

    dataset_name: str = "voicebank"
    train_manifest: str = ""
    val_manifest: str = ""
    test_manifest: str = ""

    model_name: str = "DilatedMaskCNN"
    loss_name: str = "L1"

    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3

    target_sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    segment_seconds: float = 2.0

    subset_fraction: float = 1.0
    max_train_items: int | None = None
    max_train_eval_items: int | None = None
    max_val_items: int | None = None
    max_test_items: int | None = None

    num_workers: int = 4
    pin_memory: bool = True

    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 1e-4

    compute_train_eval_loss: bool = True
    compute_test_loss: bool = True
    eval_mode: str = "full_chunks"
    max_test_loss_items: int | None = None

    max_files_to_eval: int | None = None
    save_example_count: int = 3
    # Kas kiek epochu skaiciuojamas train_eval_loss. 1 = kiekviena epocha,
    # 5 = kas penkta (paskutine epocha visada iskaiciuojama).
    eval_every_n_epochs: int = 1

    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"

    device: str = "auto"
    use_amp: bool = True

    # Gradientų normos ribojimas. None = išjungta.
    grad_clip_norm: float | None = 5.0

    # Learning rate scheduler (ReduceLROnPlateau ant val_loss).
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 3   # epochų be pagerejimo, po kurių LR mažinamas
    lr_scheduler_factor: float = 0.5  # LR daugiklis kai sumažinamas (pvz. 1e-3 → 5e-4)

    # Mokymo tęsimas iš checkpoint failo. CLI parametras --resume nustato šią reikšmę vykdymo metu.
    resume_from: str | None = None

    @property
    def segment_samples(self) -> int:
        return int(self.target_sample_rate * self.segment_seconds)

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _filter_known_keys(data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(ExperimentConfig)}
    data = _expand_env(data)
    return {k: v for k, v in data.items() if k in allowed}


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "experiment" in data:
        data = data["experiment"]

    return ExperimentConfig(**_filter_known_keys(data))



def load_experiment_list(path):
    from pathlib import Path as _Path
    import yaml as _yaml
    path = _Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = _yaml.safe_load(f) or {}
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])
    configs = []
    for item in experiments:
        merged = dict(defaults)
        merged.update(item)
        configs.append(ExperimentConfig(**_filter_known_keys(merged)))
    return configs


def save_config(config, path):
    from pathlib import Path as _Path
    import yaml as _yaml
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        _yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)
