from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_loss_curve(history_df, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss (mokymo metu)")
    if "train_eval_loss" in history_df.columns and history_df["train_eval_loss"].notna().any():
        plt.plot(history_df["epoch"], history_df["train_eval_loss"], label="Train-eval loss (pilni segmentai)")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss (pilni segmentai)")
    plt.xlabel("Epocha")
    plt.ylabel("Paklaida")
    plt.title("Modelio mokymo eiga")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _stft_db(wav: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    import scipy.signal

    _, _, zxx = scipy.signal.stft(wav, fs=1.0, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag = np.abs(zxx)
    return 20 * np.log10(mag + 1e-8)


def save_spectrogram_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    output_path: str | Path,
    n_fft: int,
    hop_length: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    specs = [
        ("Clean", _stft_db(clean, n_fft, hop_length)),
        ("Noisy", _stft_db(noisy, n_fft, hop_length)),
        ("Enhanced", _stft_db(enhanced, n_fft, hop_length)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (title, spec) in zip(axes, specs):
        im = ax.imshow(spec, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("Dažnis")
        fig.colorbar(im, ax=ax, format="%+2.0f dB")

    axes[-1].set_xlabel("Laiko kadras")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_waveform_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    output_path: str | Path,
    sr: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_len = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]
    t = np.arange(min_len) / sr

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, title, wav in zip(axes, ["Clean", "Noisy", "Enhanced"], [clean, noisy, enhanced]):
        ax.plot(t, wav)
        ax.set_title(title)
        ax.set_ylabel("Amplitudė")
        ax.grid(True)

    axes[-1].set_xlabel("Laikas, s")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
