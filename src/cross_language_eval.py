from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from .config import ExperimentConfig
from .models import create_model
from .train import evaluate_model


def load_eval_jobs(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("jobs", [])


def _config_from_job(job: dict) -> ExperimentConfig:
    """Iš vertinimo užduoties paima tik tuos raktus, kurie atitinka ExperimentConfig laukus."""
    allowed = {f.name for f in fields(ExperimentConfig)}
    return ExperimentConfig(**{k: v for k, v in job.items() if k in allowed})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    jobs = load_eval_jobs(args.config)

    for job in jobs:
        config = _config_from_job(job)
        # `checkpoint_path` nėra ExperimentConfig lauke, todėl skaitomas tiesiai iš užduoties aprašo.
        if "checkpoint_path" not in job:
            raise KeyError(
                f"cross_language job '{config.experiment_id}' neturi 'checkpoint_path' lauko."
            )
        checkpoint_path = Path(job["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Nerastas checkpoint: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(config.model_name, n_freq=config.n_freq).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        run_dir = Path(config.results_dir) / config.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

        _, summary = evaluate_model(model, config, run_dir, checkpoint_path)

        with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump({**config.to_dict(), **summary}, f, indent=2, ensure_ascii=False)

        print(f"Baigtas vertinimas: {config.experiment_id}")


if __name__ == "__main__":
    main()
