from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .audio_utils import calculate_snr_db, load_audio


def build_pairs(clean_dir: Path, noisy_dir: Path) -> list[tuple[Path, Path]]:
    clean_map = {p.name: p for p in clean_dir.glob("*.wav")}
    noisy_map = {p.name: p for p in noisy_dir.glob("*.wav")}
    common = sorted(set(clean_map) & set(noisy_map))

    if not common:
        raise RuntimeError(f"Nerasta bendrų .wav porų: {clean_dir} / {noisy_dir}")

    return [(clean_map[name], noisy_map[name]) for name in common]


def speaker_from_name(path: Path) -> str:
    # Tipinis VoiceBank formatas: p226_001.wav
    stem = path.stem
    return stem.split("_")[0]


def create_manifest_rows(pairs: list[tuple[Path, Path]], split: str, language: str, target_sr: int) -> list[dict]:
    rows = []

    for clean_path, noisy_path in pairs:
        try:
            clean = load_audio(clean_path, target_sr)
            noisy = load_audio(noisy_path, target_sr)
            snr_est = calculate_snr_db(clean, noisy)
            duration = min(len(clean), len(noisy)) / target_sr
        except Exception:
            snr_est = None
            duration = None

        rows.append(
            {
                "language": language,
                "dataset": "VoiceBank_DEMAND",
                "split": split,
                "clean_path": str(clean_path.resolve()),
                "noisy_path": str(noisy_path.resolve()),
                "noise_path": "",
                "noise_type": "unknown",
                "snr_db": snr_est,
                "snr_group": round(float(snr_est), 1) if snr_est is not None else "",
                "speaker_id": speaker_from_name(clean_path),
                "utt_id": clean_path.stem,
                "duration_sec": duration,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_clean = data_root / "clean_trainset_28spk_wav"
    train_noisy = data_root / "noisy_trainset_28spk_wav"
    test_clean = data_root / "clean_testset_wav"
    test_noisy = data_root / "noisy_testset_wav"

    train_pairs_full = build_pairs(train_clean, train_noisy)
    test_pairs = build_pairs(test_clean, test_noisy)

    # Split pagal kalbėtojus, kad validacijoje nebūtų tų pačių speakerių kiek įmanoma.
    speakers = sorted({speaker_from_name(p[0]) for p in train_pairs_full})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)

    train_speaker_count = max(1, int(len(speakers) * args.train_val_split))
    train_speakers = set(speakers[:train_speaker_count])

    train_pairs = [p for p in train_pairs_full if speaker_from_name(p[0]) in train_speakers]
    val_pairs = [p for p in train_pairs_full if speaker_from_name(p[0]) not in train_speakers]

    train_rows = create_manifest_rows(train_pairs, "train", "en", args.target_sr)
    val_rows = create_manifest_rows(val_pairs, "val", "en", args.target_sr)
    test_rows = create_manifest_rows(test_pairs, "test", "en", args.target_sr)

    pd.DataFrame(train_rows).to_csv(output_dir / "train_manifest.csv", index=False)
    pd.DataFrame(val_rows).to_csv(output_dir / "val_manifest.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_manifest.csv", index=False)

    print("VoiceBank manifestai sukurti:")
    print("Train:", len(train_rows))
    print("Val:  ", len(val_rows))
    print("Test: ", len(test_rows))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
