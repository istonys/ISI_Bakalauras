from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from .config import load_experiment_list
from .train import run_experiment


def append_failed(results_dir: Path, experiment_id: str, error: str) -> None:
    path = results_dir / "failed_runs.csv"
    df_new = pd.DataFrame([{"experiment_id": experiment_id, "error": error}])

    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML failas su experiments sarastu.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Bendras checkpoint katalogas arba prefiksas, is kurio bandyti testi "
             "kiekviena eksperimenta. Jeigu nurodyta, ieskoma <resume>/<experiment_id>.pth.",
    )
    args = parser.parse_args()

    configs = load_experiment_list(args.config)

    if not configs:
        raise RuntimeError(
            f"Config faile '{args.config}' nerasta 'experiments' saraso arba jis tuscias. "
            "run_experiments.py naudok tik su YAML, kuriame yra 'defaults' ir 'experiments' laukai."
        )

    for i, config in enumerate(configs, start=1):
        print("\n" + "=" * 80)
        print(f"Vykdomas eksperimentas {i}/{len(configs)}: {config.experiment_id}")
        print("=" * 80)

        if args.resume:
            candidate = Path(args.resume) / f"{config.experiment_id}.pth"
            if candidate.exists():
                config.resume_from = str(candidate)
                print(f"Rastas resume checkpoint: {candidate}")

        try:
            run_experiment(config)
        except Exception as exc:
            print(f"KLAIDA eksperimente {config.experiment_id}: {exc}")
            traceback.print_exc()
            append_failed(Path(config.results_dir), config.experiment_id, str(exc))
            continue


if __name__ == "__main__":
    main()
