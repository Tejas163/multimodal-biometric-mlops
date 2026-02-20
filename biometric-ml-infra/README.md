# 🔐 Biometric ML Infrastructure

Production-quality ML infrastructure for **multimodal biometric user recognition** using PyTorch, MLflow, Hydra, Ray, and PyArrow.

---

## Architecture

```
Raw biometric data
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Ray Parallel Ingestion (ingest.py)              │
│  • One Ray task per subject (CPU-parallel)       │
│  • Normalise → validate schema → write Parquet   │
└────────────────────┬─────────────────────────────┘
                     │  Parquet files (PyArrow)
                     ▼
┌──────────────────────────────────────────────────┐
│  BiometricDataset  (dataset.py)                  │
│  • Reads only requested modality columns         │
│  • Zero-fills missing modalities gracefully      │
│  • index-addressable → works with any Sampler    │
└────────────────────┬─────────────────────────────┘
                     │  {modality: Tensor} dicts
                     ▼
┌──────────────────────────────────────────────────┐
│  BiometricFusionModel  (fusion.py)               │
│                                                  │
│  face ──▶ ModalityEncoder ──┐                   │
│  finger ▶ ModalityEncoder ──┼─▶ Fusion ▶ MLP ▶ logits
│  voice ─▶ ModalityEncoder ──┘                   │
│                                                  │
│  Fusion: concat | attention | mean               │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Trainer  (trainer.py)                           │
│  • MLflow: params, metrics, checkpoints          │
│  • Early stopping + top-k checkpoint manager     │
│  • MLflow Model Registry on val improvement      │
└──────────────────────────────────────────────────┘
```

---

## Stack

| Concern | Technology | Why |
|---|---|---|
| Model | **PyTorch** | Flexible, production-proven |
| Experiment tracking | **MLflow** | Params, metrics, artifacts, registry |
| Config | **Hydra** | Composable configs + CLI overrides + multi-run sweeps |
| Data format | **PyArrow / Parquet** | Columnar, fast, schema-enforced |
| Parallelism | **Ray** | Subject-level parallel ingestion across CPU cores |
| CI | **GitHub Actions** | Lint → type-check → unit tests on every push |
| Containerisation | **Docker** | Multi-stage build for lean production images |

---

## Project Structure

```
biometric-ml-infra/
├── .github/workflows/
│   ├── ci.yml            # Lint + type-check + tests (every push/PR)
│   └── train.yml         # Data ingest + training (main branch / manual)
├── conf/                 # Hydra config tree
│   ├── config.yaml       # Root (composes all groups)
│   ├── data/biometric.yaml
│   ├── model/fusion.yaml
│   ├── training/default.yaml
│   └── mlflow/local.yaml
├── src/biometric_ml/
│   ├── data/
│   │   ├── schema.py     # PyArrow schemas for all modalities
│   │   ├── ingest.py     # Ray-parallel ingestion pipeline
│   │   ├── dataset.py    # PyTorch Dataset
│   │   └── datamodule.py # DataLoader factory
│   ├── models/
│   │   ├── encoders.py   # Per-modality encoder networks
│   │   └── fusion.py     # Late-fusion model
│   ├── training/
│   │   ├── trainer.py    # Training loop + MLflow
│   │   └── reproducibility.py
│   ├── inference/
│   │   └── pipeline.py   # Inference pipeline (registry or checkpoint)
│   └── utils/logging.py
├── scripts/
│   ├── ingest_data.py    # CLI: run ingestion
│   └── train.py          # CLI: run training
├── tests/                # pytest unit tests
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/Tejas163/biometric-ml-infra.git
cd biometric-ml-infra
pip install -e ".[dev]"
```

### 2. Ingest data

```bash
# Generates synthetic data for 200 subjects → writes Parquet splits
python scripts/ingest_data.py num_subjects=200

# Override output dir
python scripts/ingest_data.py num_subjects=500 data.parquet_dir=data/parquet_v2
```

### 3. Train

```bash
# Default training run
python scripts/train.py

# Override hyperparameters at the CLI (Hydra syntax)
python scripts/train.py training.learning_rate=5e-4 model.fusion.method=attention

# Hyperparameter sweep
python scripts/train.py --multirun \
    training.learning_rate=1e-3,5e-4,1e-4 \
    model.encoder_hidden_dim=128,256
```

### 4. View MLflow dashboard

```bash
mlflow ui --backend-store-uri mlruns/
# → open http://localhost:5000
```

### 5. Run inference

```python
from biometric_ml.inference.pipeline import InferencePipeline
import torch

# From MLflow Model Registry
pipeline = InferencePipeline.from_registry(
    model_name="BiometricFusionModel",
    stage="Production",
    tracking_uri="mlruns/",
)

# Or from a local checkpoint
pipeline = InferencePipeline.from_checkpoint(
    checkpoint_path="checkpoints/epoch_0010_loss_0.1234.pt",
    model_factory=lambda: ...,  # your model constructor
    active_modalities=["face", "fingerprint", "voice"],
)

result = pipeline.predict({
    "face":        torch.randn(512).tolist(),
    "fingerprint": torch.randn(256).tolist(),
    "voice":       torch.randn(128).tolist(),
})
print(result.top_k_ids, result.top_k_probs)
```

### 6. Run tests

```bash
pytest tests/ -v --cov=biometric_ml
```

### 7. Docker

```bash
docker build -t biometric-ml:latest .
docker run --rm biometric-ml:latest
```

---

## Configuration

All settings live in `conf/` and are composable via Hydra.

| File | Controls |
|---|---|
| `conf/data/biometric.yaml` | Modalities, feature dims, split ratios, num_workers |
| `conf/model/fusion.yaml` | Encoder dims, fusion method, MLP hidden layers, num_classes |
| `conf/training/default.yaml` | LR, epochs, batch size, seed, scheduler, early stopping |
| `conf/mlflow/local.yaml` | Tracking URI, experiment name, model registry name |

### Enable/disable modalities

```yaml
# conf/data/biometric.yaml
modalities:
  face: true
  fingerprint: true
  voice: true
  gait: false   ← set to true to include gait
```

### Switch fusion strategy

```bash
python scripts/train.py model.fusion.method=attention
```

---

## Reproducibility

Every training run is fully reproducible:

1. **Seed** — `seed_everything(seed)` seeds Python, NumPy, PyTorch CPU+CUDA.
2. **Deterministic ops** — `torch.use_deterministic_algorithms(True)`.
3. **Config snapshot** — Hydra saves the exact resolved config to `outputs/`; MLflow logs it as an artifact.
4. **Checkpoint metadata** — each `.pt` file stores `epoch`, `val_loss`, and `optimizer_state_dict`.

---

## Azure Cloud Notes

The infrastructure is Azure-ready:

- **MLflow tracking**: point `mlflow.tracking_uri` to your Azure ML MLflow endpoint.
- **CI/CD**: uncomment the Azure login + `az ml model create` steps in `train.yml`.
- **Storage**: swap local `data/parquet/` for an Azure Blob Storage mount (`abfss://`).
- **Compute**: replace the `ubuntu-latest` GitHub runner with a self-hosted Azure GPU VM.
- **Logging**: set `json_format=True` in `setup_logging()` for Azure Monitor ingestion.

---

## CI Pipeline

| Job | Trigger | Steps |
|---|---|---|
| `lint-and-typecheck` | Every push / PR | ruff lint → ruff format → mypy |
| `unit-tests` | After lint passes | pytest on Python 3.10 + 3.11 with coverage |
| `schema-validation` | After tests pass | PyArrow schema smoke test |
| `ingest-and-train` | Push to main / manual | Ingest → verify Parquet → train → upload artifacts |

---

## Design Decisions

**Why late fusion?**  
Each modality encoder learns a modality-specific representation independently before fusion. This is more robust to missing or noisy modalities than early fusion (raw concatenation), and more interpretable than intermediate fusion.

**Why PyArrow over pandas?**  
PyArrow reads only the requested columns from Parquet without loading the full file into memory — critical when each sample contains multiple large feature vectors. Schema enforcement catches data contract violations at ingest time.

**Why Ray over multiprocessing?**  
Ray's remote-task model requires zero boilerplate for cross-subject parallelism, scales from a laptop to a cluster with no code changes, and handles task failures gracefully with retries.

**Why Hydra over plain argparse?**  
Hydra enables config composition (separate files for data/model/training), CLI overrides in dot-notation, and built-in multi-run sweeps — all features that would require hundreds of lines of argparse boilerplate.
