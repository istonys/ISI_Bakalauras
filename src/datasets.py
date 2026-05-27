from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .audio_utils import get_audio_num_samples, load_audio, pad_or_crop_pair, slice_or_pad_pair
from .config import ExperimentConfig


REQUIRED_COLUMNS = {"clean_path", "noisy_path"}


def _worker_init_fn(worker_id: int) -> None:
    """Užtikrina, kad kiekvienas DataLoader workeris turėtų skirtingą, bet
    nuo torch seed priklausomą atsitiktinumo būseną."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Bendras DataLoader fabrikas su workerių seed nustatymu.

    persistent_workers=True: workeriai išlieka gyvi tarp epochų – nereikia
    perkurti procesų kiekvieną kartą (Lustre lėta).
    prefetch_factor=4: workeriai iš anksto paruošia 4 batch'us eilėje,
    kad GPU niekada nelauktų duomenų.
    """
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def load_manifest_dataframe(
    manifest_path: str | Path,
    subset_fraction: float = 1.0,
    max_items: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifestas nerastas: {manifest_path}")

    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifestui trūksta stulpelių: {missing}")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if subset_fraction < 1.0:
        n = max(1, int(len(df) * subset_fraction))
        df = df.iloc[:n].reset_index(drop=True)

    if max_items is not None:
        df = df.iloc[:max_items].reset_index(drop=True)

    return df


class RandomCropSpeechDataset(Dataset):
    """Mokymo duomenys: kiekvieną kartą iš failo paimama atsitiktinė fiksuoto ilgio atkarpa."""

    def __init__(
        self,
        manifest_path: str | Path,
        target_sr: int,
        segment_samples: int,
        subset_fraction: float = 1.0,
        max_items: int | None = None,
        seed: int = 42,
    ):
        self.df = load_manifest_dataframe(manifest_path, subset_fraction, max_items, seed)
        self.target_sr = target_sr
        self.segment_samples = segment_samples

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        clean_wav = load_audio(row["clean_path"], self.target_sr)
        noisy_wav = load_audio(row["noisy_path"], self.target_sr)

        clean_wav, noisy_wav = pad_or_crop_pair(
            clean_wav=clean_wav,
            noisy_wav=noisy_wav,
            segment_samples=self.segment_samples,
            random_crop=True,
        )

        meta = row.to_dict()
        return {
            "clean": torch.from_numpy(clean_wav).float(),
            "noisy": torch.from_numpy(noisy_wav).float(),
            "name": str(meta.get("utt_id", Path(row["clean_path"]).stem)),
            "meta": meta,
        }


class FullChunksSpeechDataset(Dataset):
    """Vertinimui: kiekvienas failas skaidomas į nuoseklius fiksuoto ilgio segmentus."""

    def __init__(
        self,
        manifest_path: str | Path,
        target_sr: int,
        segment_samples: int,
        subset_fraction: float = 1.0,
        max_items: int | None = None,
        seed: int = 42,
    ):
        self.df = load_manifest_dataframe(manifest_path, subset_fraction, max_items, seed)
        self.target_sr = target_sr
        self.segment_samples = segment_samples
        self.items: list[tuple[int, int, int, int]] = []

        for row_idx, row in self.df.iterrows():
            total_samples = self._estimate_num_samples(row)
            chunk_count = max(1, math.ceil(total_samples / self.segment_samples))
            for chunk_idx in range(chunk_count):
                start_sample = chunk_idx * self.segment_samples
                self.items.append((row_idx, start_sample, chunk_idx, chunk_count))

    def _estimate_num_samples(self, row: pd.Series) -> int:
        duration = row.get("duration_sec", None)
        if duration is not None and not pd.isna(duration):
            return max(1, int(float(duration) * self.target_sr))
        return get_audio_num_samples(row["clean_path"], self.target_sr)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row_idx, start_sample, chunk_idx, chunk_count = self.items[idx]
        row = self.df.iloc[row_idx]

        clean_wav = load_audio(row["clean_path"], self.target_sr)
        noisy_wav = load_audio(row["noisy_path"], self.target_sr)

        clean_wav, noisy_wav = slice_or_pad_pair(
            clean_wav=clean_wav,
            noisy_wav=noisy_wav,
            segment_samples=self.segment_samples,
            start_sample=start_sample,
        )

        meta = row.to_dict()
        meta.update(
            {
                "chunk_index": chunk_idx,
                "chunk_count": chunk_count,
                "chunk_start_sample": start_sample,
                "eval_mode": "full_chunks",
            }
        )
        utt_id = str(meta.get("utt_id", Path(row["clean_path"]).stem))

        return {
            "clean": torch.from_numpy(clean_wav).float(),
            "noisy": torch.from_numpy(noisy_wav).float(),
            "name": f"{utt_id}_chunk_{chunk_idx:04d}",
            "meta": meta,
        }


class ManifestRowsDataset(Dataset):
    """Failų eilutės galutiniam PESQ/STOI vertinimui; audio pilnai įkeliamas evaluate_model funkcijoje."""

    def __init__(
        self,
        manifest_path: str | Path,
        subset_fraction: float = 1.0,
        max_items: int | None = None,
        seed: int = 42,
    ):
        self.df = load_manifest_dataframe(manifest_path, subset_fraction, max_items, seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        meta = row.to_dict()
        return {
            "name": str(meta.get("utt_id", Path(row["clean_path"]).stem)),
            "meta": meta,
        }


def create_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader | None, DataLoader, DataLoader]:
    train_dataset = RandomCropSpeechDataset(
        manifest_path=config.train_manifest,
        target_sr=config.target_sample_rate,
        segment_samples=config.segment_samples,
        subset_fraction=config.subset_fraction,
        max_items=config.max_train_items,
        seed=config.seed,
    )

    train_eval_dataset = None
    if config.compute_train_eval_loss:
        train_eval_dataset = FullChunksSpeechDataset(
            manifest_path=config.train_manifest,
            target_sr=config.target_sample_rate,
            segment_samples=config.segment_samples,
            subset_fraction=config.subset_fraction,
            max_items=config.max_train_eval_items if config.max_train_eval_items is not None else config.max_train_items,
            seed=config.seed,
        )

    val_dataset = FullChunksSpeechDataset(
        manifest_path=config.val_manifest,
        target_sr=config.target_sample_rate,
        segment_samples=config.segment_samples,
        subset_fraction=config.subset_fraction,
        max_items=config.max_val_items,
        seed=config.seed + 1,
    )

    test_dataset = ManifestRowsDataset(
        manifest_path=config.test_manifest,
        subset_fraction=1.0,
        max_items=config.max_test_items,
        seed=config.seed + 2,
    )

    pin_memory = config.pin_memory and torch.cuda.is_available()
    bs = config.batch_size
    nw = config.num_workers

    train_loader = _make_loader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin_memory)

    train_eval_loader = None
    if train_eval_dataset is not None:
        train_eval_loader = _make_loader(
            train_eval_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin_memory,
        )

    val_loader = _make_loader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin_memory)

    # Testavimui PESQ/STOI vertinimui naudojame batch_size=1, num_workers=0,
    # nes audio įkraunamas evaluate_model funkcijoje pilnais failais.
    test_loader = _make_loader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin_memory)

    return train_loader, train_eval_loader, val_loader, test_loader


def create_test_loss_loader(config: ExperimentConfig) -> DataLoader:
    """Testavimo paklaidai: test failai skaidomi į nuoseklius fiksuoto ilgio segmentus."""
    test_loss_dataset = FullChunksSpeechDataset(
        manifest_path=config.test_manifest,
        target_sr=config.target_sample_rate,
        segment_samples=config.segment_samples,
        subset_fraction=1.0,
        max_items=config.max_test_loss_items if config.max_test_loss_items is not None else config.max_test_items,
        seed=config.seed + 3,
    )

    pin_memory = config.pin_memory and torch.cuda.is_available()
    return _make_loader(
        test_loss_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
