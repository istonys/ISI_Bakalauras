from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def base_defaults(args) -> dict:
    return {
        "dataset_name": "voicebank",
        "train_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/train_manifest.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/val_manifest.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/test_manifest.csv",
        "model_name": args.model_name,
        "loss_name": args.loss_name,
        "seed": args.seed,
        "epochs": args.epochs_small,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "target_sample_rate": 16000,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "segment_seconds": args.segment_seconds,
        "subset_fraction": 0.10,
        "max_files_to_eval": 100,
        "save_example_count": 2,
        "num_workers": args.num_workers,
        "early_stopping": True,
        "patience": 3,
        "min_delta": 0.0001,
        # Parametrų paieškoje train/test vertinimas išjungiamas dėl vykdymo greičio.
        # Pilnuose eksperimentuose šis vertinimas vėl įjungiamas.
        "compute_train_eval_loss": False,
        "compute_test_loss": False,
        "eval_mode": "full_chunks",
        "results_dir": "results",
        "checkpoints_dir": "checkpoints",
        "device": "auto",
        "use_amp": True,
        "use_lr_scheduler": True,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        "grad_clip_norm": 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sugeneruoja 2 etapo ir pilnu eksperimentu YAML pagal pasirinkta modeli."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--loss_name", default="L1", choices=["L1", "L2", "MSE"])
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs_small", type=int, default=10)
    parser.add_argument("--epochs_full", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--output_dir", default="configs")
    parser.add_argument("--run_tag", default="", help="Pridedamas prie experiment_id, pvz. _v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    defaults = base_defaults(args)
    experiments = []
    for lr, tag in [(0.001, "1e3"), (0.0005, "5e4"), (0.0001, "1e4")]:
        experiments.append({"experiment_id": f"param_{args.model_name.lower()}_lr_{tag}", "learning_rate": lr})
    for loss in ["L1", "L2"]:
        experiments.append({"experiment_id": f"param_{args.model_name.lower()}_loss_{loss.lower()}", "loss_name": loss})
    for nfft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        experiments.append({
            "experiment_id": f"param_{args.model_name.lower()}_stft_{nfft}_{hop}",
            "n_fft": nfft, "hop_length": hop, "win_length": nfft,
        })
    for seg, batch in [(1.0, args.batch_size), (2.0, args.batch_size), (4.0, max(1, args.batch_size // 2))]:
        experiments.append({
            "experiment_id": f"param_{args.model_name.lower()}_segment_{str(seg).replace('.', '')}s",
            "segment_seconds": seg, "batch_size": batch,
        })
    for frac in [0.10, 0.25, 0.50]:
        experiments.append({
            "experiment_id": f"param_{args.model_name.lower()}_data_{int(frac * 100)}pct",
            "subset_fraction": frac,
        })

    write_yaml(
        output_dir / "small_02_parameter_sweep_selected.yaml",
        {"defaults": defaults, "experiments": experiments},
    )

    # Pilni eksperimentai: train/test eval ijungti, pilnas dataset, visi failai.
    full_base = defaults.copy()
    full_base.update({
        "epochs": args.epochs_full,
        "subset_fraction": 1.0,
        "max_files_to_eval": None,
        "save_example_count": 3,
        "patience": 10,
        "eval_every_n_epochs": 5,
        "compute_train_eval_loss": True,
        "compute_test_loss": True,
    })

    # --- Exp B: Full VoiceBank ---
    voicebank = full_base.copy()
    voicebank.update({
        "experiment_id": f"full_voicebank_{args.model_name.lower()}{args.run_tag}",
        "dataset_name": "voicebank",
        "train_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/train_manifest.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/val_manifest.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/test_manifest.csv",
    })
    write_yaml(output_dir / "full_voicebank_selected.yaml", {"experiment": voicebank})

    # --- Exp C: LIEPA matched (~7h, suderintas su VoiceBank dydziais) ---
    liepa = full_base.copy()
    liepa.update({
        "experiment_id": f"full_liepa_{args.model_name.lower()}{args.run_tag}",
        "dataset_name": "liepa",
        "train_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/train_manifest_matched.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/val_manifest_matched.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/test_manifest_matched.csv",
    })
    write_yaml(output_dir / "full_liepa_selected.yaml", {"experiment": liepa})

    # --- Exp D: Full LIEPA (~67h) ---
    liepa_full = full_base.copy()
    liepa_full.update({
        "experiment_id": f"full_liepa_full_{args.model_name.lower()}{args.run_tag}",
        "dataset_name": "liepa",
        "epochs": 60,
        "train_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/train_manifest.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/val_manifest.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/test_manifest.csv",
    })
    write_yaml(output_dir / "full_liepa_full_selected.yaml", {"experiment": liepa_full})

    # --- Cross-language eval ---
    cross_en_on_lt = full_base.copy()
    cross_en_on_lt.update({
        "experiment_id": f"cross_en_model_on_lt_{args.model_name.lower()}{args.run_tag}",
        "dataset_name": "liepa",
        "train_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/train_manifest_matched.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/val_manifest_matched.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/LIEPA_DEMAND/test_manifest_matched.csv",
        "checkpoint_path": f"checkpoints/full_voicebank_{args.model_name.lower()}_best.pth",
        "max_files_to_eval": None,
        "save_example_count": 3,
    })

    cross_lt_on_en = full_base.copy()
    cross_lt_on_en.update({
        "experiment_id": f"cross_lt_model_on_en_{args.model_name.lower()}{args.run_tag}",
        "dataset_name": "voicebank",
        "train_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/train_manifest.csv",
        "val_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/val_manifest.csv",
        "test_manifest": "/scratch/lustre/home/${USER}/data/voicebank_28spk_manifest/test_manifest.csv",
        "checkpoint_path": f"checkpoints/full_liepa_{args.model_name.lower()}_best.pth",
        "max_files_to_eval": None,
        "save_example_count": 3,
    })

    write_yaml(output_dir / "cross_language_selected.yaml", {"jobs": [cross_en_on_lt, cross_lt_on_en]})

    print("Sugeneruota:")
    print(output_dir / "small_02_parameter_sweep_selected.yaml")
    print(output_dir / "full_voicebank_selected.yaml")
    print(output_dir / "full_liepa_selected.yaml")
    print(output_dir / "full_liepa_full_selected.yaml")
    print(output_dir / "cross_language_selected.yaml")


if __name__ == "__main__":
    main()
