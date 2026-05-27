from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ---------------------------------------------------------------------------
# Pagalbines funkcijos
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifestas nerastas: {path}")
    df = pd.read_csv(path)
    return df


def _check_columns(df: pd.DataFrame, split: str, required: list[str]) -> list[str]:
    errors = []
    for col in required:
        if col not in df.columns:
            errors.append(f"[{split}] Truksta stulpelio: '{col}'")
    return errors


def _check_file_existence(df: pd.DataFrame, split: str, cols: list[str]) -> tuple[list[str], int]:
    """Patikrina, ar visi nurodyti keliai egzistuoja. Grązina klaidas ir trukstamu failu sk."""
    errors: list[str] = []
    missing_count = 0
    for col in cols:
        if col not in df.columns:
            continue
        for path_str in df[col].dropna():
            p = Path(path_str)
            if not p.exists():
                missing_count += 1
                if missing_count <= 5:
                    errors.append(f"[{split}] Nerastas failas ({col}): {path_str}")
    if missing_count > 5:
        errors.append(f"[{split}] ... ir dar {missing_count - 5} trukstamu failu ({'+'.join(cols)})")
    return errors, missing_count


def _load_wav_fast(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


def _calculate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    noise = noisy - clean
    signal_power = float(np.mean(clean ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    return 10.0 * math.log10(signal_power / noise_power)


def _check_audio_sample(
    row: pd.Series,
    split: str,
    target_sr: int,
) -> tuple[list[str], list[str]]:
    """Tikrina viena clean/noisy pora. Grąžina (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    clean_path = row.get("clean_path", "")
    noisy_path = row.get("noisy_path", "")

    if not Path(clean_path).exists() or not Path(noisy_path).exists():
        return errors, warnings  # egzistavimo patikra jau atlikta anksčiau

    try:
        clean, clean_sr = _load_wav_fast(clean_path)
        noisy, noisy_sr = _load_wav_fast(noisy_path)
    except Exception as exc:
        errors.append(f"[{split}] Nepavyko užkrauti audio: {exc}")
        return errors, warnings

    # Sample rate
    if clean_sr != target_sr:
        errors.append(f"[{split}] clean_path SR={clean_sr}, tikimasi {target_sr}: {clean_path}")
    if noisy_sr != target_sr:
        errors.append(f"[{split}] noisy_path SR={noisy_sr}, tikimasi {target_sr}: {noisy_path}")

    # Ilgis
    if len(clean) == 0:
        errors.append(f"[{split}] Tuščias clean failas: {clean_path}")
        return errors, warnings
    if len(noisy) == 0:
        errors.append(f"[{split}] Tuščias noisy failas: {noisy_path}")
        return errors, warnings
    if abs(len(clean) - len(noisy)) > target_sr * 0.01:  # >10ms skirtumas
        warnings.append(
            f"[{split}] clean/noisy ilgiai skiriasi: "
            f"clean={len(clean)}, noisy={len(noisy)} ({clean_path})"
        )

    # NaN / inf
    if not np.all(np.isfinite(clean)):
        errors.append(f"[{split}] NaN/Inf rasta clean: {clean_path}")
    if not np.all(np.isfinite(noisy)):
        errors.append(f"[{split}] NaN/Inf rasta noisy: {noisy_path}")

    return errors, warnings


def _check_snr_sample(
    row: pd.Series,
    split: str,
    snr_tol_db: float = 1.0,
) -> tuple[list[str], list[str], float | None]:
    """Perskaičiuoja SNR ir palygina su manifesto reikšme. Grąžina (errors, warnings, computed_snr)."""
    errors: list[str] = []
    warnings: list[str] = []

    clean_path = row.get("clean_path", "")
    noisy_path = row.get("noisy_path", "")
    expected_snr = row.get("snr_db")

    if not Path(clean_path).exists() or not Path(noisy_path).exists():
        return errors, warnings, None
    if expected_snr is None or (isinstance(expected_snr, float) and math.isnan(expected_snr)):
        return errors, warnings, None

    try:
        clean, _ = _load_wav_fast(clean_path)
        noisy, _ = _load_wav_fast(noisy_path)
    except Exception as exc:
        errors.append(f"[{split}] SNR skaičiavimo klaida ({exc}): {clean_path}")
        return errors, warnings, None

    if len(clean) == 0 or len(noisy) == 0:
        return errors, warnings, None

    computed = _calculate_snr_db(clean, noisy)
    diff = abs(computed - float(expected_snr))
    if diff > snr_tol_db:
        warnings.append(
            f"[{split}] SNR neatitikimas: laukta={float(expected_snr):.1f}dB, "
            f"gauta={computed:.1f}dB (skirtumas={diff:.2f}dB): {clean_path}"
        )

    return errors, warnings, computed


def _check_speaker_overlap(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Tikrina, ar speakerijai nesikartoja tarp splitų."""
    errors = []
    if "speaker_id" not in next(iter(dfs.values())).columns:
        return ["speaker_id stulpelis nerastas — speakerių patikra praleista"]

    split_speakers: dict[str, set] = {
        split: set(df["speaker_id"].dropna().unique())
        for split, df in dfs.items()
    }
    splits = list(split_speakers.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = split_speakers[s1] & split_speakers[s2]
            if overlap:
                errors.append(
                    f"Speakeriai persidengia tarp '{s1}' ir '{s2}': "
                    f"{len(overlap)} speakeriai — {sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
                )
    return errors


def _print_summary(dfs: dict[str, pd.DataFrame], target_sr: int) -> None:
    """Atspausdina statistine suvestine."""
    print("\n" + "=" * 72)
    print("DATASET SUVESTINE")
    print("=" * 72)

    for split, df in dfs.items():
        n_files = len(df)
        if "duration_sec" in df.columns:
            hours = df["duration_sec"].sum() / 3600.0
        else:
            hours = float("nan")

        n_speakers = df["speaker_id"].nunique() if "speaker_id" in df.columns else "?"

        print(f"\n[{split.upper()}]")
        print(f"  Failai:     {n_files:,}")
        print(f"  Valandos:   {hours:.2f} h")
        print(f"  Speakeriai: {n_speakers}")

        if "snr_db" in df.columns:
            snr_counts = df["snr_db"].value_counts().sort_index()
            print(f"  SNR dB:     " + ", ".join(f"{v:.1f}({c})" for v, c in snr_counts.items()))

        if "noise_type" in df.columns:
            noise_counts = df["noise_type"].value_counts().sort_index()
            print("  Noise tipai (" + str(len(noise_counts)) + "):")
            for nt, cnt in noise_counts.items():
                print(f"    {nt:<20} {cnt:>6} failai")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Pagrindine logika
# ---------------------------------------------------------------------------

def validate(
    manifest_dir: Path,
    target_sr: int = 16000,
    snr_sample_n: int = 30,
    audio_sample_n: int = 50,
    snr_tol_db: float = 1.0,
    quick: bool = False,
    seed: int = 42,
) -> bool:
    """
    Atlieka pilna validacija. Grąžina True, jei kritiniu klaidų nėra.
    Meta RuntimeError, jei randama kritiniu klaidų.
    """
    rng = random.Random(seed)
    splits = ["train", "val", "test"]
    required_cols = ["clean_path", "noisy_path", "noise_path", "snr_db", "speaker_id", "duration_sec"]

    # --- Manifest krovimas ---
    print(f"\nManifest katalogas: {manifest_dir}")
    dfs: dict[str, pd.DataFrame] = {}
    for split in splits:
        p = manifest_dir / f"{split}_manifest.csv"
        if not p.exists():
            raise FileNotFoundError(f"Nerastas manifestas: {p}")
        dfs[split] = _load_manifest(p)
        print(f"  [{split}] {len(dfs[split]):,} eilučių įkeltos.")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # --- A) Stulpelių patikra ---
    print("\n[A] Stulpelių patikra...")
    for split, df in dfs.items():
        errs = _check_columns(df, split, required_cols)
        all_errors.extend(errs)
    if not all_errors:
        print("    OK")

    # --- B) Failų egzistavimas ---
    print("[B] Failų egzistavimo patikra...")
    total_missing = 0
    for split, df in dfs.items():
        errs, missing = _check_file_existence(df, split, ["clean_path", "noisy_path", "noise_path"])
        all_errors.extend(errs)
        total_missing += missing
    if total_missing == 0:
        print("    OK — visi failai egzistuoja")
    else:
        print(f"    KLAIDA: {total_missing} trūkstamų failų")

    if quick:
        print("\n[--quick] Audio patikra praleista.")
        _print_summary(dfs, target_sr)
        errs_speaker = _check_speaker_overlap(dfs)
        all_errors.extend(errs_speaker)
        _report_and_raise(all_errors, all_warnings)
        return True

    # --- C) Garso parametrų patikra (atsitiktinė imtis) ---
    print(f"[C] Garso parametrų patikra (atsitiktinė imtis ~{audio_sample_n} porų/split)...")
    for split, df in dfs.items():
        if len(df) == 0:
            continue
        sample_rows = df.sample(min(audio_sample_n, len(df)), random_state=rng.randint(0, 9999))
        split_errs, split_warns = 0, 0
        for _, row in sample_rows.iterrows():
            errs, warns = _check_audio_sample(row, split, target_sr)
            all_errors.extend(errs)
            all_warnings.extend(warns)
            split_errs += len(errs)
            split_warns += len(warns)
        status = "OK" if split_errs == 0 else f"KLAIDOS: {split_errs}"
        print(f"    [{split}] {status}" + (f", perspėjimai: {split_warns}" if split_warns else ""))

    # --- D) SNR perskaičiavimas ---
    print(f"[D] SNR perskaičiavimo patikra (atsitiktinė imtis ~{snr_sample_n} porų/split)...")
    snr_diffs: list[float] = []
    for split, df in dfs.items():
        if len(df) == 0:
            continue
        sample_rows = df.sample(min(snr_sample_n, len(df)), random_state=rng.randint(0, 9999))
        computed_snrs = []
        for _, row in sample_rows.iterrows():
            errs, warns, computed = _check_snr_sample(row, split, snr_tol_db)
            all_errors.extend(errs)
            all_warnings.extend(warns)
            if computed is not None:
                computed_snrs.append(computed)
                expected = float(row.get("snr_db", 0))
                snr_diffs.append(abs(computed - expected))
        if computed_snrs:
            mean_snr = sum(computed_snrs) / len(computed_snrs)
            max_diff = max(snr_diffs) if snr_diffs else 0.0
            warn_count = sum(1 for w in all_warnings if f"[{split}] SNR" in w)
            status = "OK" if warn_count == 0 else f"Perspėjimai: {warn_count}"
            print(f"    [{split}] vidutinis SNR={mean_snr:.2f}dB, max skirtumas={max_diff:.2f}dB — {status}")

    # --- E) Speaker split patikra ---
    print("[E] Speakerių persidengimo patikra...")
    errs_speaker = _check_speaker_overlap(dfs)
    all_errors.extend(errs_speaker)
    if not errs_speaker:
        speakers_total = sum(df["speaker_id"].nunique() for df in dfs.values() if "speaker_id" in df.columns)
        print(f"    OK — {speakers_total} unikalūs speakeriai, nėra persidengimo")
    else:
        print(f"    KLAIDA: {len(errs_speaker)} persidengimų")

    # --- Statistinė suvestinė ---
    _print_summary(dfs, target_sr)

    _report_and_raise(all_errors, all_warnings)
    return True


def _report_and_raise(errors: list[str], warnings: list[str]) -> None:
    if warnings:
        print(f"\nPerspėjimai ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  ⚠  {w}")
        if len(warnings) > 20:
            print(f"  ... ir dar {len(warnings) - 20} perspėjimų")

    if errors:
        print(f"\nKRITINĖS KLAIDOS ({len(errors)}):")
        for e in errors:
            print(f"  ✗  {e}")
        raise RuntimeError(
            f"Validacija nepraėjo: {len(errors)} kritinė(-ės) klaida(-os). "
            "Pataisykite dataset prieš paleidžiant eksperimentus."
        )
    else:
        print("\nValidacija PRAĖJO. Dataset yra tinkamas eksperimentams.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIEPA + DEMAND dataset validatorius. "
                    "Tikrina manifest failus ir WAV irasus prieš paleidžiant eksperimentus."
    )
    parser.add_argument(
        "--manifest_dir", required=True,
        help="Katalogas su train_manifest.csv, val_manifest.csv, test_manifest.csv",
    )
    parser.add_argument("--target_sr", type=int, default=16000, help="Tikimasi sample rate (default: 16000)")
    parser.add_argument(
        "--snr_sample_n", type=int, default=30,
        help="Kiek atsitiktinių porų patikrinti SNR (per split, default: 30)",
    )
    parser.add_argument(
        "--audio_sample_n", type=int, default=50,
        help="Kiek atsitiktinių porų patikrinti garso parametrus (per split, default: 50)",
    )
    parser.add_argument(
        "--snr_tol_db", type=float, default=1.0,
        help="Leistinas SNR skirtumas dB (default: 1.0)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Greita patikra: tik failų egzistavimas ir statistika (be garso krovimo)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate(
        manifest_dir=Path(args.manifest_dir),
        target_sr=args.target_sr,
        snr_sample_n=args.snr_sample_n,
        audio_sample_n=args.audio_sample_n,
        snr_tol_db=args.snr_tol_db,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
