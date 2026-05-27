from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .audio_utils import load_audio, mix_clean_with_noise, save_audio

ALL_DEMAND_NOISES_16K = [
    "DKITCHEN", "DLIVING", "DWASHING",
    "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "OOFFICE",
    "PCAFETER", "PRESTO", "PSTATION",
    "SPSQUARE", "STRAFFIC",
    "TBUS", "TCAR", "TMETRO",
]

# Voicebank-DEMAND stiliaus disjunktinis split'as, jei kas norėtu jį panaudoti
# kaip palyginamąjį skirstymo principą.
VOICEBANK_STYLE_TRAIN_NOISES = [
    "DKITCHEN", "DLIVING", "DWASHING",
    "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "OOFFICE",
    "PCAFETER", "PRESTO", "PSTATION",
]
VOICEBANK_STYLE_TEST_NOISES = [
    "SPSQUARE", "STRAFFIC",
    "TBUS", "TCAR", "TMETRO",
]


def _duration_seconds(path: Path) -> float:
    """Greitai grazina garso ilgi sekundemis."""
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def find_liepa_sentence_files(liepa_root: Path) -> list[Path]:
    """Surenka VISUS .wav failus is LIEPA speakeriu katalogu (D<skaitm>),
    nepriklausomai nuo sekcijos tipo (S<skaitm> = sakiniai, Z<skaitm> = zodziai/frazes).
    Z katalogai dazniausiai turi trumpus zodzius, bet kai kurie speakeriai
    juose turi ir ilgesnius irasus, todel itraukiame visa.
    Per trumpus failus (<250ms) atsiranda papildomas filtras irasymo metu.
    """
    files = list(liepa_root.rglob("D*/*/*.wav"))
    files = [
        f for f in files
        if f.parent.parent.name.startswith("D") and f.parent.parent.name[1:].isdigit()
    ]
    return sorted(files)


def find_demand_noise_files(
    demand_root: Path,
    *,
    noise_types: list[str],
    channel: str = "ch01",
) -> list[Path]:
    """Suranda po viena triuksmo kanala iš nurodytu DEMAND aplinku.

    DEMAND yra organizuotas kaip <demand_root>/<NOISE_TYPE>/ch01.wav .. ch16.wav.
    Vienkanaliam triuksmo salinimui pakanka vieno kanalo, todel imamas tik
    ch01 (numatytas) is kiekvieno triuksmo tipo.
    """
    files: list[Path] = []
    missing: list[str] = []
    for noise_type in noise_types:
        candidate = demand_root / noise_type / f"{channel}.wav"
        if candidate.exists():
            files.append(candidate)
        else:
            missing.append(noise_type)
    if missing:
        print(f"[demand] Ispejimas: nerasti triuksmai (praleisti): {missing}")
    return files


def speaker_id_from_liepa_path(path: Path) -> str:
    # Tiketina struktura: D02/S001/file.wav
    try:
        return path.parent.parent.name
    except Exception:
        return "unknown"


def split_by_speaker(files: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    speakers = sorted({speaker_id_from_liepa_path(p) for p in files})
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_train = max(1, int(len(speakers) * train_ratio))
    n_val = max(1, int(len(speakers) * val_ratio))

    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train:n_train + n_val])
    test_spk = set(speakers[n_train + n_val:])

    if not test_spk:
        # Jei per mazai speakeriu, bent keli failai bus testui pagal failu splita.
        test_spk = set(list(val_spk)[-1:])
        val_spk = set(list(val_spk)[:-1]) or test_spk

    return {
        "train": [p for p in files if speaker_id_from_liepa_path(p) in train_spk],
        "val": [p for p in files if speaker_id_from_liepa_path(p) in val_spk],
        "test": [p for p in files if speaker_id_from_liepa_path(p) in test_spk],
    }


def limit_by_hours(files: list[Path], target_sr: int, hours: float | None, seed: int) -> list[Path]:
    """Atrenka tiek failu, kad bendras ilgis pasiektu nurodyta valandu kieki."""
    if hours is None or hours <= 0:
        return files

    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)

    selected: list[Path] = []
    total_sec = 0.0
    target_sec = hours * 3600.0

    for path in files:
        total_sec += _duration_seconds(path)
        selected.append(path)
        if total_sec >= target_sec:
            break

    return sorted(selected)


def create_rows_for_split(
    split: str,
    clean_files: list[Path],
    noise_files: list[Path],
    output_dir: Path,
    target_sr: int,
    snr_values: list[float],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    clean_out_dir = output_dir / "clean" / split
    noisy_out_dir = output_dir / "noisy" / split

    if not noise_files:
        raise RuntimeError(f"[{split}] Nera triuksmo failu - patikrinkite noise type rinkini.")

    for idx, clean_path in enumerate(tqdm(clean_files, desc=f"{split} generavimas")):
        snr_db = snr_values[idx % len(snr_values)]
        noise_path = rng.choice(noise_files)

        clean = load_audio(clean_path, target_sr)
        noise = load_audio(noise_path, target_sr)

        if len(clean) < target_sr // 4:
            continue

        clean, noisy = mix_clean_with_noise(clean, noise, snr_db=snr_db)

        speaker_id = speaker_id_from_liepa_path(clean_path)
        utt_id = f"{speaker_id}_{clean_path.parent.name}_{clean_path.stem}_snr{str(snr_db).replace('.', 'p')}"

        clean_out = clean_out_dir / f"{utt_id}.wav"
        noisy_out = noisy_out_dir / f"{utt_id}.wav"

        save_audio(clean_out, clean, target_sr)
        save_audio(noisy_out, noisy, target_sr)

        rows.append({
            "language": "lt",
            "dataset": "LIEPA_DEMAND",
            "split": split,
            "clean_path": str(clean_out.resolve()),
            "noisy_path": str(noisy_out.resolve()),
            "noise_path": str(noise_path.resolve()),
            "noise_type": noise_path.parent.name,
            "snr_db": snr_db,
            "snr_group": snr_db,
            "speaker_id": speaker_id,
            "utt_id": utt_id,
            "duration_sec": len(clean) / target_sr,
            "source_clean_path": str(clean_path.resolve()),
        })

    return rows


def _parse_noise_list(s: str | None, default: list[str]) -> list[str]:
    if not s:
        return list(default)
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liepa_root", required=True)
    parser.add_argument("--demand_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument(
        "--matched_hours", type=float, default=0.0,
        help="Bendra LIEPA valandu apimtis. 0 reiskia naudoti viska.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demand_channel", default="ch01",
        help="Kuris DEMAND mikrofono kanalas naudojamas (ch01..ch16). Vienkanaliam triuksmo salinimui pakanka vieno.",
    )
    parser.add_argument(
        "--train_noises", default=None,
        help="Kableliu atskirtas triuksmu tipu sarasas mokymui/validavimui. "
             "Default: visi 17 DEMAND tipu (16k versijos).",
    )
    parser.add_argument(
        "--test_noises", default=None,
        help="Kableliu atskirtas triuksmu tipu sarasas testavimui. "
             "Default: visi 17 DEMAND tipu (toks pats kaip train).",
    )
    args = parser.parse_args()

    liepa_root = Path(args.liepa_root)
    demand_root = Path(args.demand_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_noises = _parse_noise_list(args.train_noises, ALL_DEMAND_NOISES_16K)
    test_noises = _parse_noise_list(args.test_noises, ALL_DEMAND_NOISES_16K)
    overlap = set(train_noises) & set(test_noises)
    if overlap and (args.train_noises or args.test_noises):
        # Jei triukšmų sąrašai nurodyti rankiniu būdu ir persidengia, pateikiamas įspėjimas, bet vykdymas nestabdomas.
        print(
            f"[demand] Pastaba: train ir test triuksmu rinkiniai persidengia "
            f"({len(overlap)} tipai). Tai gali buti tycinis sprendimas; jei norite "
            f"Voicebank-DEMAND stiliaus disjunktinio splito, peržiūrėkite "
            f"VOICEBANK_STYLE_TRAIN_NOISES ir VOICEBANK_STYLE_TEST_NOISES konstantas."
        )

    print(f"[demand] Train/val triuksmai ({len(train_noises)}): {train_noises}")
    print(f"[demand] Test triuksmai ({len(test_noises)}): {test_noises}")

    train_val_noise_files = find_demand_noise_files(demand_root, noise_types=train_noises, channel=args.demand_channel)
    test_noise_files = find_demand_noise_files(demand_root, noise_types=test_noises, channel=args.demand_channel)

    clean_files = find_liepa_sentence_files(liepa_root)
    if not clean_files:
        raise RuntimeError(f"Nerasta LIEPA S001/S002 .wav failu: {liepa_root}")
    if not train_val_noise_files or not test_noise_files:
        raise RuntimeError("Nerasta DEMAND triuksmu failu - patikrinkite katalogu strukutra ir kanalo varda.")

    matched_hours = args.matched_hours if args.matched_hours > 0 else None
    clean_files = limit_by_hours(clean_files, args.target_sr, matched_hours, args.seed)

    splits = split_by_speaker(clean_files, args.train_ratio, args.val_ratio, args.seed)

    snr_by_split = {
        "train": [0.0, 5.0, 10.0, 15.0],
        "val": [2.5, 7.5, 12.5, 17.5],
        "test": [2.5, 7.5, 12.5, 17.5],
    }
    noises_by_split = {
        "train": train_val_noise_files,
        "val": train_val_noise_files,
        "test": test_noise_files,
    }

    all_rows = {}
    for split, files in splits.items():
        rows = create_rows_for_split(
            split=split,
            clean_files=files,
            noise_files=noises_by_split[split],
            output_dir=output_dir,
            target_sr=args.target_sr,
            snr_values=snr_by_split[split],
            seed=args.seed + len(split),
        )
        all_rows[split] = rows
        pd.DataFrame(rows).to_csv(output_dir / f"{split}_manifest.csv", index=False)

    print("LIEPA + DEMAND paruosta:")
    for split, rows in all_rows.items():
        hours = sum(row["duration_sec"] for row in rows) / 3600.0
        print(f"{split}: {len(rows)} failu, {hours:.2f} val.")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
