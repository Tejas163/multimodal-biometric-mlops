"""
trainer.py
----------
Training loop — PyTorch port of the Kaggle notebook training setup.

Notebook training config:
    optimizer : Adam(lr=1e-4)
    loss      : categorical_crossentropy → CrossEntropyLoss
    metrics   : accuracy
    batch_size: 8
    epochs    : 50

We add on top:
    - MLflow experiment tracking
    - Checkpointing (save best model)
    - Early stopping
    - Top-1 and Top-5 accuracy
    - Learning rate scheduling
"""

from __future__ import annotations

import heapq
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.reproducibility import seed_everything

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, save_top_k: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self._heap: list[tuple[float, str]] = []

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, val_loss: float, extra: dict[str, Any] | None = None) -> Path:
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}_loss_{val_loss:.4f}.pt"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            **(extra or {}),
        }, ckpt_path)
        heapq.heappush(self._heap, (-val_loss, str(ckpt_path)))
        if len(self._heap) > self.save_top_k:
            _, worst = heapq.heappop(self._heap)
            Path(worst).unlink(missing_ok=True)
        return ckpt_path

    @property
    def best_checkpoint(self) -> Path | None:
        if not self._heap:
            return None
        return Path(min(self._heap, key=lambda x: x[0])[1])


class Trainer:
    """
    Orchestrates training, validation, checkpointing, and MLflow logging.
    Mirrors the Kaggle notebook training setup but in PyTorch + MLflow.
    """

    def __init__(self, model: BiometricFusionModel, cfg: DictConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device) -> None:
        self.model        = model.to(device)
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        t_cfg = cfg.training

        # Notebook uses: Adam(learning_rate=1e-4), categorical_crossentropy
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),  # skip frozen MobileNet
            lr=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
        )
        self.scheduler    = CosineAnnealingLR(self.optimizer, T_max=t_cfg.epochs)
        self.stopper      = EarlyStopper(patience=t_cfg.early_stopping_patience)
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=Path(t_cfg.checkpoint.dir),
            save_top_k=t_cfg.checkpoint.save_top_k,
        )
        self.grad_clip = t_cfg.gradient_clip_val

    def fit(self) -> None:
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run():
            self._log_params()
            best_val_loss = float("inf")

            try:
                for epoch in range(1, self.cfg.training.epochs + 1):
                    t0         = time.perf_counter()
                    train_loss = self._train_epoch()
                    val_loss, top1, top5 = self._val_epoch()
                    elapsed    = time.perf_counter() - t0

                    current_lr = self.scheduler.get_last_lr()[0]
                    self.scheduler.step()

                    mlflow.log_metrics({
                        "train/loss":          train_loss,
                        "val/loss":            val_loss,
                        "val/accuracy_top1":   top1,
                        "val/accuracy_top5":   top5,
                        "lr":                  current_lr,
                        "epoch_time_sec":      elapsed,
                    }, step=epoch)

                    log.info(
                        "Epoch %03d | train_loss=%.4f | val_loss=%.4f "
                        "| top1=%.3f | top5=%.3f | lr=%.2e | %.1fs",
                        epoch, train_loss, val_loss, top1, top5, current_lr, elapsed,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt = self.ckpt_manager.save(
                            self.model, self.optimizer, epoch, val_loss
                        )
                        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
                        if self.cfg.mlflow.log_model_on_best:
                            self._register_model()

                    if self.stopper.step(val_loss):
                        log.info("Early stopping at epoch %d", epoch)
                        break

            except KeyboardInterrupt:
                log.warning("Interrupted — saving emergency checkpoint")
                self.ckpt_manager.save(self.model, self.optimizer, -1, float("inf"))

            log.info("Training finished. Best val loss: %.4f", best_val_loss)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            inputs, labels = self._unpack(batch)
            labels = labels.squeeze().long()
            self.optimizer.zero_grad(set_to_none=True)
            logits: Tensor = self.model(inputs)
            loss: Tensor   = self.criterion(logits, labels)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct1   = 0
        correct5   = 0
        total      = 0

        for batch in self.val_loader:
            inputs, labels = self._unpack(batch)
            labels = labels.squeeze().long()
            logits = self.model(inputs)
            total_loss += self.criterion(logits, labels).item()

            preds = logits.argmax(dim=-1)
            correct1 += (preds == labels).sum().item()

            k = min(5, logits.size(-1))
            top_k = logits.topk(k, dim=-1).indices
            correct5 += (top_k == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

        n     = max(len(self.val_loader), 1)
        top1  = correct1 / total if total > 0 else 0.0
        top5  = correct5 / total if total > 0 else 0.0
        return total_loss / n, top1, top5

    def _unpack(self, batch: dict[str, Tensor | int]) -> tuple[dict[str, Tensor], Tensor]:
        labels = batch["label"].to(self.device)
        inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if k != "label" and isinstance(v, Tensor)
        }
        return inputs, labels

    def _log_params(self) -> None:
        flat = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)

        def _flatten(d: dict, prefix: str = "") -> dict:
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(_flatten(v, f"{key}."))
                else:
                    out[key] = v
            return out

        mlflow.log_params(_flatten(flat))
        if self.cfg.mlflow.log_config_snapshot:
            p = Path("outputs/config_snapshot.yaml")
            p.parent.mkdir(exist_ok=True)
            OmegaConf.save(self.cfg, p)
            mlflow.log_artifact(str(p), artifact_path="config")

    def _register_model(self) -> None:
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="model",
            registered_model_name=self.cfg.mlflow.registered_model_name,
        )
        log.info("Model registered as '%s'", self.cfg.mlflow.registered_model_name)


def build_and_fit(cfg: DictConfig) -> None:
    """Entry point called from scripts/train.py."""
    from biometric_ml.data.datamodule import BiometricDataModule, DataConfig

    seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)
    
    train_transforms = {
        "fingerprint": transforms.Compose([
            transforms.RandomRotation(15),      # Handle slight finger tilts
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle sensor lighting
            transforms.RandomResizedCrop(128, scale=(0.9, 1.0)), # Handle minor shifts
        ]),
        "iris": transforms.Compose([
            transforms.RandomRotation(10),      # Iris captures can have slight head tilts
            transforms.ColorJitter(brightness=0.1),
            # Grayscale images like Iris usually need fewer color transforms
        ])
    }

    # Data
    data_cfg = DataConfig(
        parquet_dir=cfg.data.parquet_dir,
        active_modalities=["fingerprint", "iris_left", "iris_right"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transforms=train_transforms
    )
    dm          = BiometricDataModule(data_cfg)
    num_classes = dm.num_classes
    log.info("Subjects (classes): %d", num_classes)

    # Model — exact notebook architecture
    model = BiometricFusionModel(num_classes=num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Parameters — trainable: %d / total: %d (MobileNet frozen)", trainable, total)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    trainer = Trainer(
        model=model, cfg=cfg,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        device=device,
    )
    trainer.fit()
