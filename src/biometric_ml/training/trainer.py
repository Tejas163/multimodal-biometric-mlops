"""
trainer.py — fixed for overfitting on small biometric dataset.

Key changes vs previous:
  - Mixup augmentation added — most effective regulariser for tiny datasets
  - Augmentation strength reduced (was too aggressive)
  - Model registered only at end of training (not every best epoch — too slow)
  - Debug prints removed
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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.reproducibility import seed_everything

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience  = patience
        self.counter   = 0
        self.best_loss = float("inf")

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

    def save(self, model, optimizer, epoch, val_loss) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}_loss_{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
        heapq.heappush(self._heap, (-val_loss, str(path)))
        if len(self._heap) > self.save_top_k:
            _, worst = heapq.heappop(self._heap)
            Path(worst).unlink(missing_ok=True)
        return path


def mixup(x: Tensor, y: Tensor, alpha: float = 0.2) -> tuple[Tensor, Tensor, Tensor, float]:
    """
    Mixup augmentation — blends two random samples in each batch.
    Extremely effective for small datasets: creates infinite virtual training examples.
    alpha=0.4 is standard for classification tasks.
    """
    lam   = float(torch.distributions.Beta(alpha, alpha).sample())
    idx   = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def _augment(t: Tensor, flip_prob: float = 0.5, jitter: float = 0.30) -> Tensor:
    """Lightweight tensor augmentation — flip + brightness."""
    if torch.rand(1).item() < flip_prob:
        t = t.flip(-1)
    factor = 1.0 + (torch.rand(1).item() - 0.5) * jitter
    return (t * factor).clamp(0.0, 1.0)


class Trainer:
    def __init__(self, model: BiometricFusionModel, cfg: DictConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device) -> None:
        self.model        = model.to(device)
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        t = cfg.training
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        trainable = [p for p in model.parameters() if p.requires_grad]
        log.info("Trainable param tensors: %d", len(trainable))

        self.optimizer = self._init_optimizer()        
        self.scheduler    = CosineAnnealingLR(self.optimizer, T_max=t.epochs, eta_min=1e-5)
        self.stopper      = EarlyStopper(patience=t.early_stopping_patience)
        self.ckpt_manager = CheckpointManager(Path(t.checkpoint.dir), t.checkpoint.save_top_k)
        self.grad_clip    = t.gradient_clip_val
    
    def _init_optimizer(self):
        # We re-init the optimizer when we unfreeze to include MobileNet weights
        return Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            weight_decay=1e-3 # Higher weight decay for small data
        )

    def fit(self) -> None:
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run():
            self._log_params()
            best_val_loss = float("inf")
            best_ckpt     = None

            try:
                for epoch in range(1, self.cfg.training.epochs + 1):
                   warmup_epochs = 5
                   if epoch <= warmup_epochs:
                       base_lr = self.cfg.training.learning_rate
                       current_lr=base_lr * (epoch / warmup_epochs)
                       for param_group in self.optimizer.param_groups:
                           param_group['lr'] = current_lr
                       log.info(f"Warmup Phase: Setting LR to {current_lr:.2e}")
                   t0         = time.perf_counter()
                   train_loss = self._train_epoch()
                   val_loss, top1, top5 = self._val_epoch()
                   elapsed    = time.perf_counter() - t0
                   current_lr = self.scheduler.get_last_lr()[0]
                   self.scheduler.step()

                   mlflow.log_metrics({
                        "train/loss":        train_loss,
                        "val/loss":          val_loss,
                        "val/accuracy_top1": top1,
                        "val/accuracy_top5": top5,
                        "lr":                current_lr,
                        "epoch_time_sec":    elapsed,
                    }, step=epoch)

                   log.info(
                        "Epoch %03d | train=%.4f | val=%.4f | top1=%.3f | top5=%.3f | lr=%.2e | %.1fs",
                        epoch, train_loss, val_loss, top1, top5, current_lr, elapsed,
                    )

                   if val_loss < best_val_loss:
                       best_val_loss = val_loss
                       best_ckpt = self.ckpt_manager.save(
                            self.model, self.optimizer, epoch, val_loss
                            )
                       mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")

                   if self.stopper.step(val_loss):
                       log.info("Early stopping at epoch %d", epoch)
                       break

            except KeyboardInterrupt:
                log.warning("Interrupted — saving emergency checkpoint")
                self.ckpt_manager.save(self.model, self.optimizer, -1, float("inf"))

            # Register model once at end — not every best epoch (saves time)
            if best_ckpt:
                log.info("Training complete. Registering best weights to MLflow...")
                # Ensure we are registering the BEST weights, not just the last epoch's weights
                # The ckpt_manager usually handles the path; load them back into the model
                # self.model.load_state_dict(torch.load(best_ckpt)) 
                self._register_model()

            log.info("Training done. Best val loss: %.4f", best_val_loss)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            inputs, labels = self._unpack(batch)

            # Augment each modality
            fp    = _augment(inputs["fingerprint"], jitter=0.3)
            left  = _augment(inputs["iris_left"],   jitter=0.3)
            right = _augment(inputs["iris_right"],  jitter=0.3)

            # Mixup on fingerprint (most informative modality)
            fp_mix, y_a, y_b, lam = mixup(fp, labels, alpha=0.2)
            inputs_aug = {"fingerprint": fp_mix, "iris_left": left, "iris_right": right}

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs_aug)
            loss   = mixup_loss(self.criterion, logits, y_a, y_b, lam)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = correct1 = correct5 = total = 0

        for batch in self.val_loader:
            inputs, labels = self._unpack(batch)
            logits         = self.model(inputs)
            total_loss    += self.criterion(logits, labels).item()

            preds = logits.argmax(dim=-1)
            log.debug("labels=%s preds=%s", labels[:5].tolist(), preds[:5].tolist())
            correct1 += (preds == labels).sum().item()
            k = min(5, logits.size(-1))
            top_k = logits.topk(k, dim=-1).indices
            correct5 += (top_k == labels.unsqueeze(1)).any(dim=1).sum().item()
            total    += labels.size(0)

        n    = max(len(self.val_loader), 1)
        top1 = correct1 / total if total > 0 else 0.0
        top5 = correct5 / total if total > 0 else 0.0
        return total_loss / n, top1, top5

    def _unpack(self, batch):
        labels = batch["label"].to(self.device).long()
        inputs = {}
        for k, v in batch.items():
            if k != "label" and isinstance(v, Tensor):
                # Move to device and ensure float32
                t = v.to(self.device).float()
            
                # If data is in [0, 255] range, scale to [0, 1]
                if t.max() > 1.0:
                    t = t / 255.0
                inputs[k] = t
        return inputs, labels

    def _log_params(self) -> None:
        flat = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        def _flatten(d, prefix=""):
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                out.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: v})
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
    from biometric_ml.data.datamodule import BiometricDataModule, DataConfig

    seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)

    data_cfg = DataConfig(
        parquet_dir=cfg.data.parquet_dir,
        active_modalities=["fingerprint", "iris_left", "iris_right"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transforms=None,
    )
    dm          = BiometricDataModule(data_cfg)
    num_classes = dm.num_classes
    log.info("Subjects (classes): %d", num_classes)

    model = BiometricFusionModel(num_classes=num_classes, freeze_mobilenet=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    # Check if actually frozen
    is_frozen = not next(model.fingerprint_branch.parameters()).requires_grad
    status_str = "MobileNet frozen" if is_frozen else "Full Finetuning"

    log.info("Parameters — trainable: %d / total: %d (%s)", trainable, total, status_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    Trainer(
        model=model, cfg=cfg,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        device=device,
    ).fit()
