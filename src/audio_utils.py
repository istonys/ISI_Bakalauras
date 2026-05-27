from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    wav, orig_sr = sf.read(str(path), always_2d=False)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if orig_sr != target_sr:
        wav = resample_audio(wav, orig_sr, target_sr)

    return wav.astype(np.float32)


def get_audio_num_samples(path: str | Path, target_sr: int) -> int:
    """Grąžina apytikslį įrašo imčių skaičių po resampling į target_sr."""
    info = sf.info(str(path))
    if info.samplerate == target_sr:
        return int(info.frames)
    return int(round(info.frames * target_sr / info.samplerate))


def save_audio(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(str(path), wav, sample_rate)


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    out = resample_poly(wav, up, down)
    return out.astype(np.float32)


def align_signals(*signals: np.ndarray) -> list[np.ndarray]:
    min_len = min(len(x) for x in signals)
    return [x[:min_len].astype(np.float32) for x in signals]


def slice_or_pad_pair(
    clean_wav: np.ndarray,
    noisy_wav: np.ndarray,
    segment_samples: int,
    start_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Iškerpa poros segmentą nuo start_sample; trumpesnį segmentą papildo nuliais."""
    clean_wav, noisy_wav = align_signals(clean_wav, noisy_wav)
    start_sample = max(0, int(start_sample))
    end_sample = start_sample + segment_samples

    clean_segment = clean_wav[start_sample:end_sample]
    noisy_segment = noisy_wav[start_sample:end_sample]

    if len(clean_segment) < segment_samples:
        pad_amount = segment_samples - len(clean_segment)
        clean_segment = np.pad(clean_segment, (0, pad_amount))
        noisy_segment = np.pad(noisy_segment, (0, pad_amount))

    return clean_segment.astype(np.float32), noisy_segment.astype(np.float32)


def pad_or_crop_pair(
    clean_wav: np.ndarray,
    noisy_wav: np.ndarray,
    segment_samples: int,
    random_crop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    clean_wav, noisy_wav = align_signals(clean_wav, noisy_wav)
    length = len(clean_wav)

    if length >= segment_samples:
        start = np.random.randint(0, length - segment_samples + 1) if random_crop else 0
        return slice_or_pad_pair(clean_wav, noisy_wav, segment_samples, start)

    return slice_or_pad_pair(clean_wav, noisy_wav, segment_samples, 0)


def compute_stft_features(
    waveforms: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stft = torch.stft(
        waveforms,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    log_mag = torch.log1p(mag)
    return stft, mag, phase, log_mag


def log_mag_pair(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grąžina švaraus ir triukšmingo signalų logaritminius amplitudės spektrus."""
    _, _, _, clean_log_mag = compute_stft_features(clean, n_fft, hop_length, win_length, window)
    _, _, _, noisy_log_mag = compute_stft_features(noisy, n_fft, hop_length, win_length, window)
    return clean_log_mag, noisy_log_mag


def reconstruct_waveform(
    pred_log_mag: torch.Tensor,
    noisy_phase: torch.Tensor,
    target_length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    pred_mag = torch.expm1(pred_log_mag).clamp(min=0.0)
    pred_stft = torch.polar(pred_mag, noisy_phase)

    return torch.istft(
        pred_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=target_length,
    )


def calculate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean, noisy = align_signals(clean, noisy)
    noise = noisy - clean
    signal_power = float(np.mean(clean ** 2) + 1e-12)
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    return 10.0 * math.log10(signal_power / noise_power)


def mix_clean_with_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    if len(noise) < len(clean):
        repeat_count = int(np.ceil(len(clean) / max(len(noise), 1)))
        noise = np.tile(noise, repeat_count)

    if len(noise) > len(clean):
        max_start = len(noise) - len(clean)
        start = random.randint(0, max_start) if max_start > 0 else 0
        noise = noise[start:start + len(clean)]

    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    target_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_power / noise_power)

    noisy = clean + scale * noise
    peak = np.max(np.abs(noisy)) + 1e-12
    if peak > 0.99:
        noisy = noisy / peak * 0.99
        clean = clean / peak * 0.99

    return clean.astype(np.float32), noisy.astype(np.float32)
