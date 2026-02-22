"""trainer.py — With class weights, augmentation for both modalities."""

from __future__ import annotations

import heapq
import logging
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from biometric_ml.data.datamodule import BiometricDataModule, DataConfig
from biometric_ml.data.dataset import BiometricDataset
from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.reproducibility import seed_everything

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    def __init__(self, d: Path, k: int):
        d.mkdir(parents=True, exist_ok=True)
        self.d = d
        self.k = k
        self.heap = []

    def save(self, model, opt, epoch, loss) -> Path:
        p = self.d / f"epoch_{epoch:04d}_loss_{loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": loss
        }, p)
        heapq.heappush(self.heap, (-loss, str(p)))
        if len(self.heap) > self.k:
            _, w = heapq.heappop(self.heap)
            Path(w).unlink(missing_ok=True)
        return p


def _augment_image(t: Tensor, flip_prob=0.5, jitter=0.2, rotate_deg=10):
    """Heavy augmentation for both modalities."""
    # Random horizontal flip
    if torch.rand(1) < flip_prob:
        t = t.flip(-1)

    # Random rotation (simplified – random shift)
    # For simplicity, we'll just apply a random affine transform? Instead, we'll use random cropping + resize.
    # But cropping would change size; we'll keep it simple for now: intensity jitter and maybe Gaussian noise.

    # Intensity jitter
    t = t * (1 + (torch.rand(1) - 0.5) * jitter)

    # Add small Gaussian noise
    if torch.rand(1) < 0.3:
        noise = torch.randn_like(t) * 0.02
        t = t + noise

    return t.clamp(0, 1)


class Trainer:
    def __init__(self, model, cfg, train_loader, val_loader, device, class_weights=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        t = cfg.training

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights)

        # Single optimizer — all params trained at same LR
        # (backbone_params split was broken: looked for 'fingerprint_features'
        #  but model uses 'fingerprint_branch')
        all_params = [p for p in model.parameters() if p.requires_grad]
        log.info("Trainable param tensors: %d", len(all_params))

        self.optimizer = Adam(
            all_params,
            lr=t.learning_rate,
            weight_decay=t.weight_decay
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t.epochs, eta_min=1e-6)
        self.stopper = EarlyStopper(t.early_stopping_patience)
        self.ckpt_manager = CheckpointManager(Path(t.checkpoint.dir), t.checkpoint.save_top_k)
        self.grad_clip = t.gradient_clip_val

    def fit(self):
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run():
            self._log_params()
            best_loss = float("inf")
            best_ckpt = None

            try:
                for epoch in range(1, self.cfg.training.epochs + 1):
                    t0 = time.perf_counter()
                    tr_loss, grad_norm = self._train()
                    val_loss, top1, top5 = self._val()
                    elapsed = time.perf_counter() - t0
                    lr = self.optimizer.param_groups[0]["lr"]

                    self.scheduler.step()

                    mlflow.log_metrics({
                        "train/loss": tr_loss,
                        "val/loss": val_loss,
                        "val/top1": top1,
                        "val/top5": top5,
                        "lr": lr,
                        "grad_norm": grad_norm
                    }, step=epoch)

                    log.info(
                        "Epoch %03d | train=%.4f | val=%.4f | "
                        "top1=%.3f | top5=%.3f | lr=%.2e | grad=%.2e | %.1fs",
                        epoch, tr_loss, val_loss, top1, top5, lr, grad_norm, elapsed
                    )

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_ckpt = self.ckpt_manager.save(
                            self.model, self.optimizer, epoch, val_loss
                        )

                    if self.stopper.step(val_loss):
                        log.info("Early stopping at epoch %d", epoch)
                        break

            except KeyboardInterrupt:
                log.warning("Interrupted")

            if self.cfg.mlflow.log_model_on_best and best_ckpt:
                mlflow.pytorch.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name=self.cfg.mlflow.registered_model_name
                )
            log.info("Done. Best val loss: %.4f", best_loss)

    def _train(self):
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inp, lbl = self._unpack(batch)

            # Debug: print input range first batch of first epoch
            if num_batches == 0 and not hasattr(self, '_printed_range'):
                print("\n[DEBUG] Input ranges:")
                for k, v in inp.items():
                    print(f"  {k}: min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")
                self._printed_range = True

            # Apply augmentation
            inp["fingerprint"] = _augment_image(inp["fingerprint"])
            inp["iris_left"] = _augment_image(inp["iris_left"])
            inp["iris_right"] = _augment_image(inp["iris_right"])

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(inp)
            loss = self.criterion(logits, lbl)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Compute total gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            total_grad_norm += total_norm

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches, total_grad_norm / num_batches

    @torch.no_grad()
    def _val(self):
        self.model.eval()
        total_loss = correct_1 = correct_5 = num_samples = 0
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            inp, lbl = self._unpack(batch)
            logits = self.model(inp)
            total_loss += self.criterion(logits, lbl).item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbl.cpu().tolist())
            correct_1 += (preds == lbl).sum().item()
            k = min(5, logits.size(-1))
            correct_5 += (logits.topk(k, -1).indices == lbl.unsqueeze(1)).any(1).sum().item()
            num_samples += lbl.size(0)

        # Show which subjects the model gets right
        correct_subjects = [l for l, p in zip(all_labels, all_preds) if l == p]
        log.debug("Val correct subjects: %s / %d total", sorted(correct_subjects), num_samples)
        if correct_1 == 0:
            # Show closest misses (label in top-3 preds)
            near_misses = sum(1 for l, p in zip(all_labels, all_preds) if abs(l - p) <= 2)
            log.debug("Near misses (|label-pred|<=2): %d", near_misses)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        return avg_loss, correct_1 / max(num_samples, 1), correct_5 / max(num_samples, 1)

    def _unpack(self, batch):
        lbl = batch["label"].to(self.device).long()
        inp = {
            k: v.to(self.device).float()
            for k, v in batch.items()
            if k in ("fingerprint", "iris_left", "iris_right")
        }
        return inp, lbl

    def _log_params(self):
        flat = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)

        def flatten(d, prefix=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out.update(flatten(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out

        mlflow.log_params(flatten(flat))


# ----------------------------------------------------------------------
# Helper to compute class weights
# ----------------------------------------------------------------------
def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for balanced loss.
    For classes with zero samples, weight is set to 1.0.
    For present classes, weight = total_samples / (num_classes * class_count).
    """
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    total = counts.sum()
    weights = torch.ones(num_classes, dtype=torch.float32)  # default 1.0 for missing classes
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights


# ----------------------------------------------------------------------
# Top‑level training functions
# ----------------------------------------------------------------------
def build_and_fit(cfg: DictConfig) -> None:
    """Regular training with fixed train/val/test splits."""
    seed_everything(cfg.training.seed, cfg.training.deterministic)

    dm = BiometricDataModule(DataConfig(
        parquet_dir=cfg.data.parquet_dir,
        active_modalities=["fingerprint", "iris_left", "iris_right"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transforms=None,
    ))

    num_classes = dm.num_classes
    log.info("Subjects (classes): %d", num_classes)

    # Get training dataset from the dataloader (safe, always indexable)
    train_loader = dm.train_dataloader()
    train_dataset = train_loader.dataset
    class_weights = compute_class_weights(train_dataset, num_classes)
    log.info("Class weights (first 10): %s", class_weights[:10].tolist())

    model = BiometricFusionModel(num_classes=num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Params trainable=%d / total=%d (%.1f%% frozen)",
             trainable, total, 100 * (1 - trainable/total))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Sanity check
    with torch.no_grad():
        batch = next(iter(dm.train_dataloader()))
        inp = {k: v[:4].to(device) for k, v in batch.items() if k != "label"}
        lbl = batch["label"][:4]

        out = model(inp)
        preds = out.argmax(dim=-1)

        log.info("Sanity check — labels: %s, preds: %s", lbl.tolist(), preds.tolist())
        unique_preds = preds.unique().numel()
        if unique_preds == 1:
            log.error("CRITICAL: Model predictions collapsed to single class!")
        else:
            log.info("Sanity check passed: %d unique predictions", unique_preds)

    Trainer(model, cfg, dm.train_dataloader(), dm.val_dataloader(), device,
            class_weights=class_weights).fit()

def build_and_fit_cv(cfg: DictConfig) -> None:
    """5-fold cross-validation training."""
    seed_everything(cfg.training.seed, cfg.training.deterministic)

    parquet_dir = Path(cfg.data.parquet_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = _create_folds(parquet_dir, n_folds=5, seed=cfg.training.seed)

    base_dataset = BiometricDataset(
        parquet_dir / "train.parquet",
        ["fingerprint", "iris_left", "iris_right"]
    )

    num_classes = base_dataset.num_classes
    log.info("Total subjects (classes): %d", num_classes)

    output_dir = Path(cfg.training.checkpoint.dir) / "cv_folds"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for fold_config in folds:
        metrics = _train_fold(
            fold_config, base_dataset, cfg, device, output_dir, num_classes
        )
        all_metrics.append(metrics)

    _log_cv_summary(all_metrics)
