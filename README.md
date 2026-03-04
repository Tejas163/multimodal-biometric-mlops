# 🔐 Biometric ML Infrastructure

[![CI](https://github.com/Tejas163/biometric-ml-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/Tejas163/biometric-ml-infra/actions/workflows/ci.yml)
[![Training](https://github.com/Tejas163/biometric-ml-infra/actions/workflows/train.yml/badge.svg)](https://github.com/Tejas163/biometric-ml-infra/actions/workflows/train.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-quality ML infrastructure for **multimodal biometric user recognition**.  
Fuses iris (left/right) and fingerprint modalities using a late-fusion PyTorch model, tracked end-to-end with MLflow, config-driven with Hydra, ingested in parallel with Ray, and stored in columnar Parquet via PyArrow.

---

## 📐 Architecture

```
Raw biometric data  (iris images / fingerprint scans)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  Ray Parallel Ingestion          (ingest.py)        │
│  • One Ray remote task per subject → CPU-parallel   │
│  • Normalise features, validate PyArrow schema      │
│  • Write split-stratified Parquet (train/val/test) │
└──────────────────────┬──────────────────────────────┘
                        │  Snappy-compressed Parquet
                        ▼
┌─────────────────────────────────────────────────────┐
│  BiometricDataset            (dataset.py)           │
│  • Reads only active-modality columns (PyArrow)     │
│  • Zero-fills NULL modalities gracefully            │
│  • Index-addressable → any PyTorch Sampler works    │
└──────────────────────┬──────────────────────────────┘
                        │  {modality: Tensor} batch dicts
                        ▼
┌─────────────────────────────────────────────────────┐
│  BiometricFusionModel        (fusion.py)            │
│                                                     │
│  fingerprint ──▶ CNN Branch ──┐                    │
│  iris_left   ──▶ CNN Branch ──┼──▶ Concat ──▶ MLP ──▶ logits
│  iris_right  ──▶ CNN Branch ──┘                    │
│                                                     │
│  Fusion strategies: concat │ attention │ mean       │
└──────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Trainer                     (trainer.py)           │
│  • MLflow: params, metrics, config snapshot         │
│  • Top-k checkpoint manager + early stopping        │
│  • Auto-registers best model in MLflow Registry     │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Concern | Technology | Rationale |
|---|---|---|
| Model | **PyTorch 2.1+** | Flexible, production-proven deep learning |
| Experiment tracking | **MLflow** | Params, metrics, artifacts, model registry |
| Config management | **Hydra** | Composable configs, CLI overrides, multi-run sweeps |
| Data format | **PyArrow / Parquet** | Columnar, schema-enforced, memory-efficient |
| Parallel ingestion | **Ray** | Zero-boilerplate subject-level parallelism |
| CI/CD | **GitHub Actions** | Lint → type-check → tests on every push |
| Containerisation | **Docker** | Multi-stage build for lean production images |

---

## 📁 Project Structure

```
biometric-ml-infra/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + type-check + pytest (every push/PR)
│       └── train.yml           # Ingest + train pipeline (main / manual trigger)
│
├── conf/                       # Hydra config tree — zero hardcoded values in src/
│   ├── config.yaml             # Root config (composes all groups)
│   ├── data/biometric.yaml     # Modalities, feature dims, split ratios
│   ├── model/fusion.yaml       # Encoder dims, fusion method, MLP layers
│   ├── training/default.yaml   # LR, epochs, batch size, seed, scheduler
│   └── mlflow/local.yaml       # Tracking URI, experiment name, registry name
│
├── src/biometric_ml/
│   ├── data/
│   │   ├── schema.py           # PyArrow schemas for all 4 modalities
│   │   ├── ingest.py           # Ray-parallel ingestion → Parquet
│   │   ├── dataset.py          # PyTorch Dataset (column-selective Parquet reads)
│   │   └── datamodule.py       # DataLoader factory + WeightedRandomSampler
│   ├── models/
│   │   ├── encoders.py         # Per-modality Linear→LayerNorm→GELU encoders
│   │   └── fusion.py           # Late-fusion model (concat/attention/mean)
│   ├── training/
│   │   ├── trainer.py          # Training loop + MLflow + checkpointing
│   │   └── reproducibility.py  # Seed everything + deterministic CUDA
│   ├── inference/
│   │   └── pipeline.py         # Load from MLflow Registry or local checkpoint
│   └── utils/
│       └── logging.py          # Structured JSON / human-readable logging
│
├── scripts/
│   ├── ingest_data.py          # CLI: run Ray ingestion pipeline
│   └── train.py                # CLI: Hydra entry point for training
│
├── tests/
│   ├── test_dataset.py         # PyArrow schema + BiometricDataset unit tests
│   ├── test_model.py           # Encoder + fusion model unit tests
│   └── test_inference.py       # InferencePipeline unit tests
│
├── Dockerfile                  # Multi-stage production container
├── requirements.txt
├── setup.py
└── pyproject.toml              # Ruff + mypy + pytest config
```

---

## ⚡ Quickstart

### Prerequisites

- Python **3.12** — [download here](https://www.python.org/downloads/release/python-3126/)
- Git

### 1. Clone & install

```bash
git clone https://github.com/Tejas163/biometric-ml-infra.git
cd biometric-ml-infra

# Create virtual environment
py -3.12 -m venv .venv

# Activate  (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Activate  (macOS / Linux)
source .venv/bin/activate

# Install package + dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Ingest data

```bash
# Generate synthetic biometric data for 200 subjects → Parquet splits
python scripts/ingest_data.py num_subjects=200

# Verify files were created
ls data/parquet/        # train.parquet  val.parquet  test.parquet
```

### 3. Train

```bash
# Quick demo run (5 epochs)
python scripts/train.py training.epochs=5

# Override hyperparameters using Hydra dot-notation
python scripts/train.py training.learning_rate=5e-4 model.fusion.method=attention

# Hyperparameter sweep
python scripts/train.py --multirun \
    training.learning_rate=1e-3,5e-4,1e-4 \
    model.fusion.method=concat,attention
```

### 4. View MLflow dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **http://localhost:5000** — every run's parameters, metrics, config snapshot, and registered model will be visible.

### 5. Run inference

```python
from biometric_ml.inference.pipeline import InferencePipeline
import torch

# Load from MLflow Model Registry
pipeline = InferencePipeline.from_registry(
    model_name="BiometricFusionModel",
    stage="Production",
    tracking_uri="sqlite:///mlflow.db",
)

# Or load from a local checkpoint
pipeline = InferencePipeline.from_checkpoint(
    checkpoint_path="checkpoints/epoch_0005_loss_1.2345.pt",
    model_factory=lambda: ...,
    active_modalities=["face", "fingerprint", "voice"],
)

result = pipeline.predict({
    "fingerprint": torch.randn(3, 128, 128).tolist(),
    "iris_left":    torch.randn(1, 64, 64).tolist(),
    "iris_right":   torch.randn(1, 64, 64).tolist(),
})

print(result.top_k_ids)    # [6, 17, 8, 2, 3]
print(result.top_k_probs)  # [0.61, 0.18, 0.09, 0.07, 0.05]
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

## ⚙️ Configuration

All settings live in `conf/` — **no values are hardcoded in source files.**

| Config file | Controls |
|---|---|
| `conf/data/biometric.yaml` | Active modalities, feature dims, split ratios, num_workers |
| `conf/model/fusion.yaml` | Encoder hidden dim, fusion method, MLP hidden layers |
| `conf/training/default.yaml` | LR, epochs, batch size, seed, scheduler, early stopping |
| `conf/mlflow/local.yaml` | Tracking URI, experiment name, model registry name |

### Enable / disable modalities

```yaml
# conf/data/biometric.yaml
modalities:
  face: true
  fingerprint: true
  voice: true
  gait: false       # set true to add gait modality
```

### Switch fusion strategy

```bash
python scripts/train.py model.fusion.method=attention
```

### Point to a remote MLflow server

```bash
python scripts/train.py mlflow.tracking_uri=https://my-mlflow.azureml.net
```

---

## 🔁 Reproducibility

Every training run is fully reproducible:

1. **Seed** — `seed_everything(seed)` seeds Python `random`, NumPy, PyTorch CPU and all CUDA devices.
2. **Deterministic ops** — `torch.use_deterministic_algorithms(True)` prevents non-deterministic CUDA kernels.
3. **Config snapshot** — Hydra saves the fully resolved config; MLflow logs it as an artifact alongside every run.
4. **Checkpoint metadata** — every `.pt` file stores `epoch`, `val_loss`, and `optimizer_state_dict`.

---

## 🔬 Supported Modalities

| Modality | Shape | Representation |
|---|---|---|
| Fingerprint | 128×128×3 (RGB) | Raw pixel tensor |
| Iris Left | 64×64×1 (grayscale) | Raw pixel tensor |
| Iris Right | 64×64×1 (grayscale) | Raw pixel tensor |

---

## 🤖 CI Pipeline

| Job | Trigger | Steps |
|---|---|---|
| `lint-and-typecheck` | Every push / PR | `ruff check` → `ruff format` → `mypy` |
| `unit-tests` | After lint | `pytest` on Python 3.11 + 3.12 with coverage |
| `schema-validation` | After tests | PyArrow schema smoke test |
| `ingest-and-train` | Push to `main` / manual | Ingest → verify Parquet → train → upload MLflow artifacts |

The `train.yml` workflow accepts manual inputs for `epochs`, `learning_rate`, and `fusion_method` from the GitHub Actions UI.

---

## ☁️ Azure Cloud Deployment

| Component | Local | Azure |
|---|---|---|
| MLflow tracking | `mlruns/` (local) | Azure ML MLflow endpoint |
| Data storage | `data/parquet/` | Azure Blob Storage (`abfss://`) |
| CI compute | `ubuntu-latest` runner | Self-hosted Azure GPU VM |
| Logging | Human-readable stdout | JSON → Azure Monitor |
| Model registry | Local MLflow | Azure ML Model Registry |

Uncomment the Azure steps in `.github/workflows/train.yml` and add `AZURE_CREDENTIALS` to your repo secrets.

---

## 🧠 Design Decisions

**Why late fusion?**
Each modality encoder learns independently before features are combined. This is more robust to missing or noisy modalities than early fusion, and allows per-modality quality inspection.

**Why PyArrow instead of pandas?**
PyArrow reads only the columns you request — disabling a modality means that data is never loaded into memory. Schema enforcement at write time catches data contract breaks at ingestion, not silently during training.

**Why Ray instead of multiprocessing?**
Ray's `@remote` decorator requires zero pool management. The same code scales from a laptop to a multi-node cluster without changes.

**Why Hydra instead of argparse?**
Config composition, dot-notation CLI overrides, and built-in multi-run sweeps — without writing any sweep logic. Config snapshots are automatic.

**Why LayerNorm instead of BatchNorm in encoders?**
LayerNorm is stable at small batch sizes and normalises correctly on zero vectors (missing modality placeholders). BatchNorm is undefined on all-zero inputs and breaks at batch size 1 during inference.

---

## 📋 Windows Notes

- **`num_workers: 0`** in `conf/data/biometric.yaml` — required on Windows; PyTorch multiprocessing uses `spawn` which requires top-level script importability.
- **`pin_memory: false`** by default — enable only when training on a CUDA GPU.
- **Python 3.14 not supported** — Ray and PyTorch do not yet have 3.14 wheels. Use Python **3.12**.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
