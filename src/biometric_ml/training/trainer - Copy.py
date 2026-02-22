"""trainer.py — clean version for 1480-row augmented dataset."""
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
from torch.utils.data import DataLoader

from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.reproducibility import seed_everything

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.counter  = 0
        self.best     = float("inf")

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
        self.d = d; self.k = k; self.heap = []

    def save(self, model, opt, epoch, loss) -> Path:
        import heapq
        p = self.d / f"epoch_{epoch:04d}_loss_{loss:.4f}.pt"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "val_loss": loss}, p)
        heapq.heappush(self.heap, (-loss, str(p)))
        if len(self.heap) > self.k:
            _, w = heapq.heappop(self.heap)
            Path(w).unlink(missing_ok=True)
        return p


def _aug(t: Tensor, flip=0.5, jitter=0.1) -> Tensor:
    if torch.rand(1) < flip:
        t = t.flip(-1)
    return (t * (1 + (torch.rand(1) - 0.5) * jitter)).clamp(0, 1)


def mixup(x, y, alpha=0.2):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class Trainer:
    def __init__(self, model, cfg, train_loader, val_loader, device):
        self.model = model.to(device)
        self.cfg   = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device = device
        t = cfg.training

        self.crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.opt  = Adam(trainable, lr=t.learning_rate, weight_decay=t.weight_decay)
        self.sched = CosineAnnealingLR(self.opt, T_max=t.epochs, eta_min=1e-6)
        self.stop  = EarlyStopper(t.early_stopping_patience)
        self.ckpt  = CheckpointManager(Path(t.checkpoint.dir), t.checkpoint.save_top_k)
        self.clip  = t.gradient_clip_val

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
                    tr = self._train()
                    vl, t1, t5 = self._val()
                    elapsed = time.perf_counter() - t0
                    lr = self.opt.param_groups[0]["lr"]
                    self.sched.step()

                    mlflow.log_metrics({"train/loss": tr, "val/loss": vl,
                                        "val/top1": t1, "val/top5": t5, "lr": lr}, step=epoch)
                    log.info("Epoch %03d | train=%.4f | val=%.4f | top1=%.3f | top5=%.3f | lr=%.2e | %.1fs",
                             epoch, tr, vl, t1, t5, lr, elapsed)

                    if vl < best_loss:
                        best_loss = vl
                        best_ckpt = self.ckpt.save(self.model, self.opt, epoch, vl)

                    if self.stop.step(vl):
                        log.info("Early stopping at epoch %d", epoch)
                        break
            except KeyboardInterrupt:
                log.warning("Interrupted")

            if self.cfg.mlflow.log_model_on_best and best_ckpt:
                mlflow.pytorch.log_model(self.model, artifact_path="model",
                    registered_model_name=self.cfg.mlflow.registered_model_name)
            log.info("Done. Best val loss: %.4f", best_loss)

    def _train(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            inp, lbl = self._unpack(batch)
            fp    = _aug(inp["fingerprint"], jitter=0.15)
            left  = _aug(inp["iris_left"],   jitter=0.05)
            right = _aug(inp["iris_right"],  jitter=0.05)
            fp_m, ya, yb, lam = mixup(fp, lbl)
            self.opt.zero_grad(set_to_none=True)
            logits = self.model({"fingerprint": fp_m, "iris_left": left, "iris_right": right})
            loss = lam * self.crit(logits, ya) + (1 - lam) * self.crit(logits, yb)
            loss.backward()
            if self.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val(self):
        self.model.eval()
        tl = c1 = c5 = n = 0
        for batch in self.val_loader:
            inp, lbl = self._unpack(batch)
            logits = self.model(inp)
            tl += self.crit(logits, lbl).item()
            preds = logits.argmax(-1)
            log.debug("labels=%s preds=%s", lbl[:5].tolist(), preds[:5].tolist())
            c1 += (preds == lbl).sum().item()
            k = min(5, logits.size(-1))
            c5 += (logits.topk(k, -1).indices == lbl.unsqueeze(1)).any(1).sum().item()
            n  += lbl.size(0)
        nb = max(len(self.val_loader), 1)
        return tl / nb, c1 / max(n, 1), c5 / max(n, 1)

    def _unpack(self, batch):
        lbl = batch["label"].to(self.device).long()
        inp = {k: v.to(self.device).float() for k, v in batch.items()
               if k != "label" and isinstance(v, Tensor)}
        return inp, lbl

    def _log_params(self):
        flat = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        def fl(d, p=""):
            out = {}
            for k, v in d.items():
                out.update(fl(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
            return out
        mlflow.log_params(fl(flat))


def build_and_fit(cfg: DictConfig) -> None:
    from biometric_ml.data.datamodule import BiometricDataModule, DataConfig
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

    model = BiometricFusionModel(num_classes=num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    log.info("Params trainable=%d / total=%d", trainable, total_p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    Trainer(model, cfg, dm.train_dataloader(), dm.val_dataloader(), device).fit()
