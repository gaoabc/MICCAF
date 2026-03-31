from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io_utils import ensure_dir, save_json
from .losses import total_loss
from .metrics import summarize_metrics
from .survival import risk_from_hazards
from .utils import detach_metrics, move_to_device


@dataclass
class EpochResult:
    losses: Dict[str, float]
    metrics: Dict[str, float]


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "epoch,split,loss_total,loss_surv,loss_ib,loss_imp,c_index,t_auc\n",
                encoding="utf-8",
            )

    def log(self, epoch: int, split: str, losses: Dict[str, float], metrics: Dict[str, float]) -> None:
        row = [
            str(epoch),
            split,
            f"{losses.get('loss_total', 0.0):.8f}",
            f"{losses.get('loss_surv', 0.0):.8f}",
            f"{losses.get('loss_ib', 0.0):.8f}",
            f"{losses.get('loss_imp', 0.0):.8f}",
            f"{metrics.get('c_index', 0.0):.8f}",
            f"{metrics.get('t_auc', 0.0):.8f}",
        ]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience


def _run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    lambda_ib: float,
    lambda_imp: float,
    train: bool,
    grad_clip_norm: float | None = None,
) -> EpochResult:
    model.train(mode=train)
    all_losses: List[Dict[str, float]] = []
    all_times: List[np.ndarray] = []
    all_events: List[np.ndarray] = []
    all_risks: List[np.ndarray] = []
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        batch = move_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            losses = total_loss(
                fused_hazards=outputs["fused_hazards"],
                bins=batch["bins"],
                events=batch["events"],
                ib_loss_value=outputs["ib_loss"],
                imp_loss_value=outputs["imp_loss"],
                lambda_ib=lambda_ib,
                lambda_imp=lambda_imp,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        risk = risk_from_hazards(outputs["fused_hazards"]).detach().cpu().numpy()
        all_times.append(batch["times"].detach().cpu().numpy())
        all_events.append(batch["events"].detach().cpu().numpy())
        all_risks.append(risk)
        all_losses.append(detach_metrics(losses))
    if all_losses:
        mean_losses = {key: float(np.mean([x[key] for x in all_losses])) for key in all_losses[0].keys()}
    else:
        mean_losses = {"loss_total": 0.0, "loss_surv": 0.0, "loss_ib": 0.0, "loss_imp": 0.0}
    times = np.concatenate(all_times) if all_times else np.array([])
    events = np.concatenate(all_events) if all_events else np.array([])
    risks = np.concatenate(all_risks) if all_risks else np.array([])
    metrics = summarize_metrics(times, events, risks) if len(times) > 0 else {"c_index": 0.0, "t_auc": 0.0}
    return EpochResult(losses=mean_losses, metrics=metrics)


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, config: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def fit_model(model, train_loader, val_loader, optimizer, device, output_dir, config) -> Dict[str, Dict[str, float]]:
    output_dir = ensure_dir(output_dir)
    logger = CSVLogger(output_dir / "history.csv")
    stopper = EarlyStopping(patience=int(config.training.early_stopping_patience), mode=str(config.training.monitor_mode))
    best_metric = None
    summary = {}
    for epoch in range(1, int(config.training.epochs) + 1):
        train_result = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_ib=float(config.loss.lambda_ib),
            lambda_imp=float(config.loss.lambda_imp),
            train=True,
            grad_clip_norm=float(config.training.grad_clip_norm),
        )
        val_result = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            lambda_ib=float(config.loss.lambda_ib),
            lambda_imp=float(config.loss.lambda_imp),
            train=False,
            grad_clip_norm=None,
        )
        logger.log(epoch, "train", train_result.losses, train_result.metrics)
        logger.log(epoch, "val", val_result.losses, val_result.metrics)
        summary[f"epoch_{epoch}"] = {
            "train_losses": train_result.losses,
            "train_metrics": train_result.metrics,
            "val_losses": val_result.losses,
            "val_metrics": val_result.metrics,
        }
        monitor_value = val_result.metrics[str(config.training.monitor)]
        if best_metric is None or ((config.training.monitor_mode == "max" and monitor_value > best_metric) or (config.training.monitor_mode == "min" and monitor_value < best_metric)):
            best_metric = monitor_value
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, dict(config))
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, dict(config))
        if stopper.step(monitor_value):
            break
    save_json(summary, output_dir / "summary.json")
    return summary


def evaluate_model(model, loader, device, config) -> Dict[str, Dict[str, float]]:
    result = _run_epoch(
        model,
        loader,
        optimizer=None,
        device=device,
        lambda_ib=float(config.loss.lambda_ib),
        lambda_imp=float(config.loss.lambda_imp),
        train=False,
        grad_clip_norm=None,
    )
    return {"losses": result.losses, "metrics": result.metrics}
