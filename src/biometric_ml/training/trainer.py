"""
trainer.py
----------
Production training loop for the BiometricFusionModel.

Responsibilities:
  * Run train / validation epochs.
  * Log all metrics, hyperparameters, and artifacts to MLflow.
  * Save checkpoints; maintain ``save_top_k`` best models.
  * Early stopping based on validation loss.
  * Register the best model in the MLflow Model Registry.
  * Gracefully handle KeyboardInterrupt (saves checkpoint on exit).

MLflow integration points:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  mlflow.start_run()                                                 │
  │    ├── log_params(config snapshot)                                  │
  │    ├── log_artifact(config.yaml)                                    │
  │    ├── per epoch: log_metrics(train/loss, val/loss, val/acc, lr)   │
  │    ├── on improvement: log_artifact(checkpoint), register_model     │
  │    └── on finish: mlflow.end_run()                                  │
  └─────────────────────────────────────────────────────────────────────┘
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.reproducibility import seed_everything

log = logging.getLogger(__name__)


class EarlyStopper:
    """Tracks validation loss and signals when training should stop."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    """
    Maintains the top-k checkpoints on disk by validation loss.

    Uses a max-heap (negated losses) to efficiently evict the worst
    checkpoint when a new best replaces it.
    """

    def __init__(self, checkpoint_dir: Path, save_top_k: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        # heap of (neg_val_loss, path) — worst on top for easy eviction
        self._heap: list[tuple[float, str]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}_loss_{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                **(extra or {}),
            },
            ckpt_path,
        )
        heapq.heappush(self._heap, (-val_loss, str(ckpt_path)))

        # Evict worst checkpoint if over budget
        if len(self._heap) > self.save_top_k:
            _, worst_path = heapq.heappop(self._heap)
            Path(worst_path).unlink(missing_ok=True)
            log.debug("Evicted checkpoint: %s", worst_path)

        return ckpt_path

    @property
    def best_checkpoint(self) -> Path | None:
        if not self._heap:
            return None
        # Best = most negative neg_val_loss = lowest val_loss
        return Path(min(self._heap, key=lambda x: x[0])[1])


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates training, validation, checkpointing, and MLflow logging.

    Args:
        model:        The BiometricFusionModel to train.
        cfg:          Full Hydra DictConfig (data + model + training + mlflow).
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        device:       torch.device to run training on.
    """

    def __init__(
        self,
        model: BiometricFusionModel,
        cfg: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        t_cfg = cfg.training
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            model.parameters(),
            lr=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.stopper = EarlyStopper(patience=t_cfg.early_stopping_patience)
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=Path(t_cfg.checkpoint.dir),
            save_top_k=t_cfg.checkpoint.save_top_k,
        )
        self.grad_clip = t_cfg.gradient_clip_val

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop inside an MLflow run."""
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run():
            self._log_params_and_config()
            best_val_loss = float("inf")

            try:
                for epoch in range(1, self.cfg.training.epochs + 1):
                    t0 = time.perf_counter()
                    train_loss = self._train_epoch(epoch)
                    val_loss, val_acc = self._val_epoch(epoch)
                    elapsed = time.perf_counter() - t0

                    current_lr = self.scheduler.get_last_lr()[0]
                    self.scheduler.step()

                    metrics = {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "lr": current_lr,
                        "epoch_time_sec": elapsed,
                    }
                    mlflow.log_metrics(metrics, step=epoch)

                    log.info(
                        "Epoch %03d | train_loss=%.4f | val_loss=%.4f "
                        "| val_acc=%.3f | lr=%.2e | %.1fs",
                        epoch, train_loss, val_loss, val_acc, current_lr, elapsed,
                    )

                    # Checkpoint + model registry on improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt_path = self.ckpt_manager.save(
                            self.model, self.optimizer, epoch, val_loss
                        )
                        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                        if self.cfg.mlflow.log_model_on_best:
                            self._register_model()

                    if self.stopper.step(val_loss):
                        log.info("Early stopping triggered at epoch %d", epoch)
                        break

            except KeyboardInterrupt:
                log.warning("Training interrupted — saving emergency checkpoint.")
                self.ckpt_manager.save(
                    self.model, self.optimizer, epoch=-1, val_loss=float("inf")
                )

            log.info("Training finished. Best val loss: %.4f", best_val_loss)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            inputs, labels = self._unpack_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits: Tensor = self.model(inputs)
            loss: Tensor = self.criterion(logits, labels)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            inputs, labels = self._unpack_batch(batch)
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def _unpack_batch(
        self, batch: dict[str, Tensor | int]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Split a dataset batch into model inputs dict and label tensor."""
        labels = batch["label"].to(self.device)
        inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if k != "label" and isinstance(v, Tensor)
        }
        return inputs, labels

    def _build_scheduler(self):
        s = self.cfg.training.scheduler
        if s.name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=s.T_max)
        if s.name == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        # "none" — constant LR; wrap in a dummy scheduler
        return CosineAnnealingLR(self.optimizer, T_max=10000)

    def _log_params_and_config(self) -> None:
        """Flatten and log all config values as MLflow params."""
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

        # Save complete config YAML as an artifact for reproducibility
        if self.cfg.mlflow.log_config_snapshot:
            config_path = Path("outputs/config_snapshot.yaml")
            config_path.parent.mkdir(exist_ok=True)
            OmegaConf.save(self.cfg, config_path)
            mlflow.log_artifact(str(config_path), artifact_path="config")

    def _register_model(self) -> None:
        """Log the model to MLflow and register it in the Model Registry."""
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="model",
            registered_model_name=self.cfg.mlflow.registered_model_name,
        )
        log.info(
            "Model registered as '%s'", self.cfg.mlflow.registered_model_name
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_and_fit(cfg: DictConfig) -> None:
    """
    Top-level entry point called from ``scripts/train.py``.

    Wires together:
      reproducibility → data → model → trainer → fit()
    """
    from biometric_ml.data.datamodule import BiometricDataModule, DataConfig
    from biometric_ml.models.fusion import BiometricFusionModel

    # 1. Reproducibility
    seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)

    # 2. Data
    active_modalities = [
        m for m, enabled in cfg.data.modalities.items() if enabled
    ]
    feature_dims = {
        m: cfg.data.feature_dims[m] for m in active_modalities
    }
    data_cfg = DataConfig(
        parquet_dir=cfg.data.parquet_dir,
        active_modalities=active_modalities,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    dm = BiometricDataModule(data_cfg)

    # Derive num_classes from the actual dataset, not the config.
    # The config value can differ from the real subject count, causing
    # "Target N is out of bounds" IndexError during training.
    num_classes = dm.num_classes
    log.info("Detected %d unique subjects (classes) in training data", num_classes)

    # 3. Model
    model = BiometricFusionModel.from_config(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        data_cfg=OmegaConf.to_container(cfg.data, resolve=True),
        num_classes=num_classes,
        feature_dims=feature_dims,
    )
    log.info(
        "Model parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # 4. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # 5. Fit
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        device=device,
    )
    trainer.fit()
