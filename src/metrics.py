from __future__ import annotations

import numpy as np
from pesq import pesq
from pystoi import stoi

from .audio_utils import align_signals, calculate_snr_db


# Per kokį trumpiausią garso ilgį (sekundėmis) PESQ/STOI dar bandoma skaičiuoti.
MIN_DURATION_SEC_FOR_METRICS = 0.25


def _sanitize(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a.astype(np.float32))
    return np.clip(a, -1.0, 1.0)


def compute_metrics_for_pair(
    clean_wav: np.ndarray,
    noisy_wav: np.ndarray,
    enhanced_wav: np.ndarray,
    sr: int,
) -> dict | None:
    """Apskaičiuoja PESQ, STOI, ir įvertintą SNR triukšmingam ir apdorotam signalui.

    Jei įrašas per trumpas arba PESQ/STOI nepavyksta – grąžinama None.
    """
    clean, noisy, enhanced = align_signals(clean_wav, noisy_wav, enhanced_wav)
    clean = _sanitize(clean)
    noisy = _sanitize(noisy)
    enhanced = _sanitize(enhanced)

    if len(clean) < int(sr * MIN_DURATION_SEC_FOR_METRICS):
        return None

    try:
        baseline_pesq = pesq(sr, clean, noisy, "wb")
        enhanced_pesq = pesq(sr, clean, enhanced, "wb")
        baseline_stoi = stoi(clean, noisy, sr, extended=False)
        enhanced_stoi = stoi(clean, enhanced, sr, extended=False)
        noisy_snr = calculate_snr_db(clean, noisy)
        enhanced_snr = calculate_snr_db(clean, enhanced)
    except Exception:
        return None

    return {
        "pesq_noisy": float(baseline_pesq),
        "pesq_enhanced": float(enhanced_pesq),
        "delta_pesq": float(enhanced_pesq - baseline_pesq),
        "stoi_noisy": float(baseline_stoi),
        "stoi_enhanced": float(enhanced_stoi),
        "delta_stoi": float(enhanced_stoi - baseline_stoi),
        "snr_noisy_est": float(noisy_snr),
        "snr_enhanced_est": float(enhanced_snr),
        "delta_snr_est": float(enhanced_snr - noisy_snr),
    }


_METRIC_KEYS = (
    "pesq_noisy",
    "pesq_enhanced",
    "delta_pesq",
    "stoi_noisy",
    "stoi_enhanced",
    "delta_stoi",
    "snr_noisy_est",
    "snr_enhanced_est",
    "delta_snr_est",
)


def summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}

    summary: dict = {"num_eval_files": len(rows)}
    for key in _METRIC_KEYS:
        vals = [row[key] for row in rows if key in row and row[key] is not None]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))
    return summary
