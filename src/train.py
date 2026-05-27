from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

# Naudojam naująją torch.amp API (PyTorch >= 2.1). Senesnėms versijoms paliekam
# fallback į torch.cuda.amp.
try:
    from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
    _AMP_NEW_API = True
except ImportError:  # pragma: no cover – senesnė PyTorch versija
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]
    _AMP_NEW_API = False

from .audio_utils import (
    align_signals,
    log_mag_pair,
    compute_stft_features,
    load_audio,
    reconstruct_waveform,
    save_audio,
)
from .config import ExperimentConfig, load_config, save_config
from .datasets import create_dataloaders, create_test_loss_loader
from .losses import create_criterion
from .metrics import compute_metrics_for_pair, summarize_metric_rows
from .models import count_parameters, create_model
from .plotting import save_loss_curve, save_spectrogram_comparison, save_waveform_comparison


# ---------------------------------------------------------------------------
# AMP helper
# ---------------------------------------------------------------------------

def _make_scaler(enabled: bool) -> GradScaler:
    """GradScaler suderinamas tiek su nauja, tiek sena torch AMP API."""
    if _AMP_NEW_API:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    """autocast() suderinamas su abiem API. Jei išjungta – grąžiname null kontekstą."""
    if not enabled:
        return nullcontext()
    if _AMP_NEW_API:
        return autocast(device_type="cuda", enabled=True)
    return autocast(enabled=True)


# ---------------------------------------------------------------------------
# Determinizmas
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Greitis svarbiau už visišką determinizmą – cudnn pasirinks tinkamiausius
    # algoritmus pirmajam batch'ui ir naudos juos toliau.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(config: ExperimentConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


# ---------------------------------------------------------------------------
# Bendra epochos eiga: ji pati tinka tiek mokymui, tiek validavimui /
# testavimui (skirtumas tik per optimizer ir scaler perdavimą).
# ---------------------------------------------------------------------------

def run_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    config: ExperimentConfig,
    window: torch.Tensor,
    device: torch.device,
    *,
    optimizer=None,
    scaler: GradScaler | None = None,
    desc: str = "Train",
    grad_clip_norm: float | None = None,
) -> float:
    """Vienos epochos eiga.

    Jei `optimizer` perduotas – modelis mokomas; jei `None` – tik vertinama.
    grad_clip_norm: jei nustatytas, gradientai apkarpomi prieš optimizer.step().
    """
    is_train = optimizer is not None
    model.train(is_train)

    use_amp = (
        is_train
        and scaler is not None
        and config.use_amp
        and device.type == "cuda"
    )

    total_loss = 0.0
    total_items = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in tqdm(loader, leave=False, desc=desc):
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)
            batch_size = clean.shape[0]

            # STFT visada skaičiuojamas fp32, nepriklausomai nuo AMP.
            with torch.no_grad():
                clean_log_mag, noisy_log_mag = log_mag_pair(
                    clean, noisy,
                    config.n_fft, config.hop_length, config.win_length, window,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _autocast(use_amp):
                pred_log_mag = model(noisy_log_mag.unsqueeze(1)).squeeze(1)
                loss = criterion(pred_log_mag, clean_log_mag)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        # unscale_ prieš clipping – reikalinga su AMP kad gradientai
                        # būtų tikrosios fp32 reikšmės, o ne masštuotos fp16.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

    return total_loss / max(total_items, 1)


def enhance_waveform(
    model: torch.nn.Module,
    noisy_wav: np.ndarray,
    config: ExperimentConfig,
    window: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Apdoroja vieną signalą per visą savo ilgį (be chunk'ų)."""
    model.eval()
    noisy_tensor = torch.from_numpy(noisy_wav).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, noisy_phase, noisy_log_mag = compute_stft_features(
            noisy_tensor, config.n_fft, config.hop_length, config.win_length, window,
        )
        pred_log_mag = model(noisy_log_mag.unsqueeze(1)).squeeze(1)
        enhanced_tensor = reconstruct_waveform(
            pred_log_mag=pred_log_mag,
            noisy_phase=noisy_phase,
            target_length=noisy_tensor.shape[1],
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=window,
        )

    return enhanced_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpointai (su resume palaikymu)
# ---------------------------------------------------------------------------

def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _load_rng_state(state: dict) -> None:
    if state is None:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    scheduler,
    config: ExperimentConfig,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    history_rows: list[dict],
    is_best: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _rng_state(),
        "config": config.to_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history_rows": history_rows,
        "num_parameters": int(count_parameters(model)),
    }
    torch.save(payload, path)
    if is_best:
        # papildomai laikom „best" atskirai – kad CSV / vertinimas visada
        # rastų geriausią modelį tame pačiame faile.
        torch.save(payload, path.with_name(path.stem + "_best.pth"))


# ---------------------------------------------------------------------------
# Mokymo eiga
# ---------------------------------------------------------------------------

def train_model(
    config: ExperimentConfig,
    run_dir: Path,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, pd.DataFrame, dict]:
    set_seed(config.seed)
    device = get_device(config)

    print(f"Naudojamas įrenginys: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, train_eval_loader, val_loader, _ = create_dataloaders(config)

    model = create_model(config.model_name, n_freq=config.n_freq).to(device)
    criterion = create_criterion(config.loss_name)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = _make_scaler(enabled=config.use_amp and device.type == "cuda")
    window = torch.hann_window(config.win_length, device=device)

    # LR scheduler: mažina learning rate kai val_loss negerėja.
    # ReduceLROnPlateau: po `patience` epochų be pagerejimo LR × factor.
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.lr_scheduler_patience,
            factor=config.lr_scheduler_factor,
            min_lr=1e-6,
        )

    # Bazinės būsenos – gali būti perrašytos, jei resume.
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict] = []

    resume_path = Path(config.resume_from) if config.resume_from else None
    if resume_path and resume_path.exists():
        print(f"Tęsiame mokymą iš checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _load_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history_rows = list(checkpoint.get("history_rows", []))

    total_train_start = time.perf_counter()
    early_stopped = False

    print("\n--- Pradedamas mokymas ---")
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_one_epoch(
            model, train_loader, criterion, config, window, device,
            optimizer=optimizer, scaler=scaler, desc="Train",
            grad_clip_norm=config.grad_clip_norm,
        )

        # Train-eval skaičiuojamas tik kas N-ąją epochą, kad pilni mokymai (50 epochų,
        # didelis duomenų kiekis) nepatirtų dviguba vertinimo kaina. Paskutinę epochą
        # visada įskaičiuojam, kad „best_train_eval_loss" būtų prasmingas.
        train_eval_loss = None
        if train_eval_loader is not None:
            n = max(1, int(config.eval_every_n_epochs or 1))
            is_last_epoch = epoch == config.epochs
            if epoch % n == 0 or is_last_epoch:
                train_eval_loss = run_one_epoch(
                    model, train_eval_loader, criterion, config, window, device,
                    desc="Train eval",
                )

        val_loss = run_one_epoch(
            model, val_loader, criterion, config, window, device,
            desc="Validation",
        )

        # Scheduler žingsnis – val_loss mažėjimo tempas kontroliuoja LR.
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss - config.min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            status = "Pagerėjo. Modelis išsaugotas."
        else:
            epochs_without_improvement += 1
            status = "Nepagerėjo."

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
            "improved": improved,
        })

        _save_checkpoint(
            checkpoint_path,
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            config=config, epoch=epoch,
            best_val_loss=best_val_loss, best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            history_rows=history_rows,
            is_best=improved,
        )

        train_eval_text = (
            f"Train-eval {config.loss_name}: {train_eval_loss:.5f} | "
            if train_eval_loss is not None else ""
        )
        print(
            f"{epoch}/{config.epochs} | "
            f"Train {config.loss_name}: {train_loss:.5f} | "
            f"{train_eval_text}"
            f"Val {config.loss_name}: {val_loss:.5f} | "
            f"LR: {current_lr:.2e} | "
            f"{status} | "
            f"{epoch_time:.1f}s"
        )

        if config.early_stopping and epochs_without_improvement >= config.patience:
            early_stopped = True
            print(f"Early stopping: {config.patience} epochas (-ų) nebuvo pagerėjimo.")
            break

    total_train_time = time.perf_counter() - total_train_start

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "history.csv", index=False)

    # Įkeliam geriausią svorį galutiniam vertinimui.
    best_path = checkpoint_path.with_name(checkpoint_path.stem + "_best.pth")
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    train_summary = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_train_eval_loss": _best_train_eval_loss(history_df, best_epoch),
        "epochs_completed": int(len(history_rows)),
        "early_stopped": bool(early_stopped),
        "train_time_total_sec": float(total_train_time),
        "train_time_total_min": float(total_train_time / 60.0),
        "avg_epoch_time_sec": float(history_df["epoch_time_sec"].mean()) if not history_df.empty else None,
        "num_parameters": int(count_parameters(model)),
    }

    return model, history_df, train_summary


def _best_train_eval_loss(history_df: pd.DataFrame, best_epoch: int) -> float | None:
    """Grąžina train_eval_loss vertę geriausios epochos eilutei, jei tokia yra."""
    if history_df.empty or "train_eval_loss" not in history_df.columns or best_epoch <= 0:
        return None
    row = history_df.loc[history_df["epoch"] == best_epoch, "train_eval_loss"]
    if row.empty:
        return None
    value = row.iloc[0]
    if pd.isna(value):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Vertinimas
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    config: ExperimentConfig,
    run_dir: Path,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict]:
    device = get_device(config)
    _, _, _, test_loader = create_dataloaders(config)

    window = torch.hann_window(config.win_length, device=device)
    model = model.to(device).eval()

    test_loss = None
    if config.compute_test_loss:
        criterion = create_criterion(config.loss_name)
        test_loss_loader = create_test_loss_loader(config)
        test_loss = run_one_epoch(
            model, test_loss_loader, criterion, config, window, device,
            desc="Test loss",
        )

    metric_rows: list[dict] = []
    examples: list[dict] = []
    skipped = 0

    eval_start = time.perf_counter()

    for idx, batch in enumerate(tqdm(test_loader, desc="Vertinimas")):
        if config.max_files_to_eval is not None and idx >= config.max_files_to_eval:
            break

        meta = _unwrap_batch_meta(batch.get("meta", {}))
        clean_path = meta.get("clean_path")
        noisy_path = meta.get("noisy_path")

        clean_wav = load_audio(clean_path, config.target_sample_rate)
        noisy_wav = load_audio(noisy_path, config.target_sample_rate)
        clean_wav, noisy_wav = align_signals(clean_wav, noisy_wav)

        enhanced_wav = enhance_waveform(
            model=model, noisy_wav=noisy_wav, config=config,
            window=window, device=device,
        )

        metrics = compute_metrics_for_pair(
            clean_wav=clean_wav, noisy_wav=noisy_wav, enhanced_wav=enhanced_wav,
            sr=config.target_sample_rate,
        )
        if metrics is None:
            skipped += 1
            continue

        row = dict(meta)
        row.update(metrics)
        row["experiment_id"] = config.experiment_id
        row["model_name"] = config.model_name
        row["loss_name"] = config.loss_name
        metric_rows.append(row)

        examples.append({
            "idx": idx,
            "clean": clean_wav,
            "noisy": noisy_wav,
            "enhanced": enhanced_wav,
            "delta_pesq": metrics["delta_pesq"],
            "delta_stoi": metrics["delta_stoi"],
            "utt_id": str(meta.get("utt_id", f"sample_{idx:04d}")),
        })

    eval_time = time.perf_counter() - eval_start

    file_metrics_df = pd.DataFrame(metric_rows)
    file_metrics_df.to_csv(run_dir / "file_metrics.csv", index=False)

    summary = summarize_metric_rows(metric_rows)
    summary["test_loss"] = float(test_loss) if test_loss is not None else None
    summary["eval_time_sec"] = float(eval_time)
    summary["eval_time_min"] = float(eval_time / 60.0)
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["num_skipped_files"] = int(skipped)

    save_examples(examples, config, run_dir)

    return file_metrics_df, summary


def _unwrap_batch_meta(meta) -> dict:
    """DataLoader su batch_size=1 paverčia kiekvieną reikšmę į list arba tensor."""
    if not isinstance(meta, dict):
        return {}
    out: dict = {}
    for key, value in meta.items():
        if isinstance(value, list):
            out[key] = value[0]
        elif torch.is_tensor(value):
            out[key] = value.item() if value.numel() == 1 else value[0].item()
        else:
            out[key] = value
    return out


def save_examples(examples: list[dict], config: ExperimentConfig, run_dir: Path) -> None:
    if not examples or config.save_example_count <= 0:
        return

    audio_dir = run_dir / "audio_examples"
    spec_dir = run_dir / "spectrograms"
    wave_dir = run_dir / "waveforms"

    sorted_best = sorted(examples, key=lambda x: x["delta_pesq"], reverse=True)
    sorted_worst = sorted(examples, key=lambda x: x["delta_pesq"])
    median_value = np.median([e["delta_pesq"] for e in examples])
    median_first = sorted(examples, key=lambda x: abs(x["delta_pesq"] - median_value))

    selected: list[dict] = []
    for label, source in [("best", sorted_best), ("typical", median_first), ("worst", sorted_worst)]:
        if source:
            item = dict(source[0])
            item["label"] = label
            selected.append(item)

    # Jei reikia daugiau pavyzdžių, papildome geriausiais (nedubliuojant).
    for item in sorted_best:
        if len(selected) >= config.save_example_count:
            break
        if all(item["idx"] != s["idx"] for s in selected):
            item = dict(item)
            item["label"] = f"extra_{len(selected) + 1}"
            selected.append(item)

    for item in selected[:config.save_example_count]:
        base = f"{item['label']}_{item['utt_id']}".replace("/", "_").replace("\\", "_")

        save_audio(audio_dir / f"{base}_clean.wav", item["clean"], config.target_sample_rate)
        save_audio(audio_dir / f"{base}_noisy.wav", item["noisy"], config.target_sample_rate)
        save_audio(audio_dir / f"{base}_enhanced.wav", item["enhanced"], config.target_sample_rate)

        save_spectrogram_comparison(
            clean=item["clean"], noisy=item["noisy"], enhanced=item["enhanced"],
            output_path=spec_dir / f"{base}_spectrogram.png",
            n_fft=config.n_fft, hop_length=config.hop_length,
        )
        save_waveform_comparison(
            clean=item["clean"], noisy=item["noisy"], enhanced=item["enhanced"],
            output_path=wave_dir / f"{base}_waveform.png",
            sr=config.target_sample_rate,
        )


# ---------------------------------------------------------------------------
# CSV agregavimas (atomiškai – saugu lygiagrečiai vykdomiems jobs)
# ---------------------------------------------------------------------------

def append_all_runs(results_dir: Path, row: dict) -> None:
    """Atomiškai prideda eilutę į results/all_runs.csv.

    Naudoja `fcntl.flock` lygiagrečiai vykdomų SLURM jobų sinchronizacijai.
    Lock failas yra atskiras, kad pasilenkimai neatsirastų skaitant CSV.
    Linux/Unix tik – Windows fallback dirba be locko.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "all_runs.csv"
    lock_path = results_dir / ".all_runs.lock"

    try:
        import fcntl
    except ImportError:  # pragma: no cover – Windows
        _merge_and_write(path, row)
        return

    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            _merge_and_write(path, row)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _merge_and_write(path: Path, row: dict) -> None:
    """Skaito esamą CSV (jei yra), prideda naują eilutę, įrašo atgal.

    Pandas užtikrina, kad visi stulpeliai (taip pat naujai atsiradę) bus
    sutvarkyti tinkamai.
    """
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    # Įrašome į laikiną failą ir atomiškai pakeičiame – sumažina riziką
    # palikti pusiau įrašytą CSV.
    tmp_path = path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Vienas eksperimentas (train + evaluate + save results)
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> dict:
    results_dir = Path(config.results_dir)
    checkpoints_dir = Path(config.checkpoints_dir)

    run_dir = results_dir / config.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"{config.experiment_id}.pth"

    save_config(config, run_dir / "config_effective.yaml")

    model, history_df, train_summary = train_model(config, run_dir, checkpoint_path)
    save_loss_curve(history_df, run_dir / "loss_curve.png")

    _, eval_summary = evaluate_model(model, config, run_dir, checkpoint_path)

    run_summary = dict(config.to_dict())
    run_summary.update(train_summary)
    run_summary.update(eval_summary)

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    append_all_runs(results_dir, run_summary)

    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Treniruoja ir vertina viena eksperimenta.")
    parser.add_argument("--config", required=True, help="Kelias iki YAML config failo.")
    parser.add_argument("--resume", default=None, help="Checkpoint failo kelias, is kurio tesi mokymasi.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config.resume_from = args.resume

    summary = run_experiment(config)

    print("\n--- Eksperimentas baigtas ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
