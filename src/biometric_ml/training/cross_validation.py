"""Cross-validation training for small biometric datasets."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from biometric_ml.data.dataset import BiometricDataset
from biometric_ml.data.schema import FUSED_SCHEMA
from biometric_ml.models.fusion import BiometricFusionModel
from biometric_ml.training.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def create_folds(parquet_dir: Path, n_folds: int = 5, seed: int = 42) -> list[dict[str, Any]]:
    """
    Create n-fold cross-validation splits from train.parquet.
    
    Returns list of dicts with train/val indices for each fold.
    """
    # Read all training data
    train_path = parquet_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"No train.parquet found in {parquet_dir}")
    
    table = pq.read_table(train_path, columns=["subject_id", "label"])
    d = table.to_pydict()
    
    subject_ids = np.array(d["subject_id"])
    labels = np.array(d["label"])
    n_samples = len(subject_ids)
    
    # Group samples by subject for stratified splitting
    unique_subjects = sorted(set(subject_ids))
    n_subjects = len(unique_subjects)
    
    log.info("Creating %d folds from %d subjects (%d samples)", 
             n_folds, n_subjects, n_samples)
    
    # Stratified split: ensure each fold has representative subjects
    rng = np.random.RandomState(seed)
    shuffled_subjects = unique_subjects[:]
    rng.shuffle(shuffled_subjects)
    
    fold_size = n_subjects // n_folds
    
    folds = []
    for fold_idx in range(n_folds):
        # Determine val subjects for this fold
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < n_folds - 1 else n_subjects
        val_subjects = set(shuffled_subjects[start_idx:end_idx])
        train_subjects = set(shuffled_subjects) - val_subjects
        
        # Get sample indices
        train_indices = [i for i, sid in enumerate(subject_ids) if sid in train_subjects]
        val_indices = [i for i, sid in enumerate(subject_ids) if sid in val_subjects]
        
        folds.append({
            "fold": fold_idx + 1,
            "train_subjects": sorted(train_subjects),
            "val_subjects": sorted(val_subjects),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
        })
        
        log.info("Fold %d: %d train samples (%d subjects), %d val samples (%d subjects)",
                 fold_idx + 1, len(train_indices), len(train_subjects),
                 len(val_indices), len(val_subjects))
    
    return folds


def train_fold(
    fold_config: dict[str, Any],
    base_dataset: BiometricDataset,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    """
    Train a single fold and return metrics.
    """
    fold_num = fold_config["fold"]
    log.info("=" * 60)
    log.info("TRAINING FOLD %d / %d", fold_num, 5)
    log.info("=" * 60)
    
    # Create subset datasets for this fold
    train_ds = Subset(base_dataset, fold_config["train_indices"])
    val_ds = Subset(base_dataset, fold_config["val_indices"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    # Get actual number of classes from full dataset
    num_classes = base_dataset.num_classes
    
    # Create fresh model for this fold
    model = BiometricFusionModel(num_classes=num_classes)
    
    # Verify freezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Fold %d: trainable=%d / total=%d (%.1f%% frozen)",
             fold_num, trainable, total, 100 * (1 - trainable/total))
    
    # Update config for this fold's checkpoint directory
    fold_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    fold_cfg.training.checkpoint.dir = str(output_dir / f"fold_{fold_num}")
    
    # Train
    trainer = Trainer(model, fold_cfg, train_loader, val_loader, device)
    
    # Monkey-patch fit to save best metrics
    best_metrics = {"val_loss": float("inf"), "top1": 0.0, "top5": 0.0}
    
    original_fit = trainer.fit
    
    def patched_fit():
        # Simplified fit that captures best metrics
        mlflow.set_tracking_uri(fold_cfg.mlflow.tracking_uri)
        mlflow.set_experiment(f"{fold_cfg.mlflow.experiment_name}_fold{fold_num}")
        
        with mlflow.start_run():
            # Log fold info
            mlflow.log_params({
                "fold": fold_num,
                "train_subjects": len(fold_config["train_subjects"]),
                "val_subjects": len(fold_config["val_subjects"]),
            })
            
            # Run training
            original_fit()
            
            # Capture final metrics (they're logged to MLflow)
            # We need to extract them from the trainer's history
            # For now, return placeholder and get from MLflow later
    
    # Actually, just run normal fit and get metrics from checkpoint
    trainer.fit()
    
    # Load best checkpoint to get final metrics
    ckpt_path = Path(fold_cfg.training.checkpoint.dir)
    best_ckpt = min(ckpt_path.glob("epoch_*.pt"), 
                    key=lambda p: float(p.stem.split("loss_")[-1].replace(".pt", "")))
    ckpt = torch.load(best_ckpt, map_location=device)
    
    metrics = {
        "fold": fold_num,
        "best_val_loss": ckpt.get("val_loss", float("nan")),
        "train_subjects": len(fold_config["train_subjects"]),
        "val_subjects": len(fold_config["val_subjects"]),
        "checkpoint": str(best_ckpt),
    }
    
    log.info("Fold %d complete: best val_loss=%.4f", fold_num, metrics["best_val_loss"])
    
    return metrics


def cross_validate(cfg: DictConfig) -> dict[str, float]:
    """
    Run k-fold cross-validation and aggregate results.
    """
    from biometric_ml.training.reproducibility import seed_everything
    
    seed_everything(cfg.training.seed, cfg.training.deterministic)
    
    parquet_dir = Path(cfg.data.parquet_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create folds
    folds = create_folds(parquet_dir, n_folds=5, seed=cfg.training.seed)
    
    # Load base dataset (all training data)
    base_dataset = BiometricDataset(
        parquet_dir / "train.parquet",
        ["fingerprint", "iris_left", "iris_right"]
    )
    
    # Create output directory for folds
    output_dir = Path(cfg.training.checkpoint.dir) / "cv_folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each fold
    all_metrics = []
    for fold_config in folds:
        metrics = train_fold(fold_config, base_dataset, cfg, device, output_dir)
        all_metrics.append(metrics)
    
    # Aggregate results
    log.info("=" * 60)
    log.info("CROSS-VALIDATION SUMMARY")
    log.info("=" * 60)
    
    avg_loss = np.mean([m["best_val_loss"] for m in all_metrics])
    std_loss = np.std([m["best_val_loss"] for m in all_metrics])
    
    log.info("Fold results:")
    for m in all_metrics:
        log.info("  Fold %d: val_loss=%.4f", m["fold"], m["best_val_loss"])
    log.info("Average val_loss: %.4f ± %.4f", avg_loss, std_loss)
    
    # Save summary
    summary = {
        "n_folds": 5,
        "avg_val_loss": float(avg_loss),
        "std_val_loss": float(std_loss),
        "fold_metrics": all_metrics,
    }
    
    import json
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary