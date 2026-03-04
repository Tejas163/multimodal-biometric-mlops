"""
scripts/demo_inference.py
--------------------------
Interactive inference demo for interviews.

Shows the full pipeline:
  1. Load a trained checkpoint
  2. Pick a random subject from the test set
  3. Load their fingerprint + iris images from disk
  4. Run inference
  5. Print a clean ranked result with confidence bars

Usage:
    python scripts/demo_inference.py

    # Use a specific checkpoint
    python scripts/demo_inference.py --checkpoint checkpoints/epoch_0042_loss_2.1234.pt

    # Test against a specific subject
    python scripts/demo_inference.py --subject 7

    # Show top-N predictions
    python scripts/demo_inference.py --top_k 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq

from biometric_ml.models.fusion import BiometricFusionModel

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"


def load_image_rgb(path: Path, size=(128, 128)) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB").resize(size), dtype=np.float32) / 255.0


def load_image_gray(path: Path, size=(64, 64)) -> np.ndarray:
    from PIL import Image

    arr = np.array(Image.open(path).convert("L").resize(size), dtype=np.float32) / 255.0
    return arr[:, :, np.newaxis]


def get_bmps(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() == ".bmp" and "desktop" not in p.name.lower()
    )


def load_subject_images(subject_dir: Path) -> dict[str, torch.Tensor]:
    """Load and preprocess one subject's images into model-ready tensors."""

    # Fingerprint: average all images → (3, 128, 128)
    fp_paths = get_bmps(subject_dir / "Fingerprint")
    if not fp_paths:
        raise FileNotFoundError(f"No fingerprint BMPs in {subject_dir / 'Fingerprint'}")
    fp_vecs = [load_image_rgb(p).flatten() for p in fp_paths]
    fp_mean = np.mean(fp_vecs, axis=0).reshape(128, 128, 3)  # H,W,C
    fp_tensor = torch.tensor(fp_mean, dtype=torch.float32).permute(2, 0, 1)  # C,H,W

    # Iris left: first available image → (1, 64, 64)
    left_paths = get_bmps(subject_dir / "left")
    if not left_paths:
        raise FileNotFoundError(f"No left iris BMPs in {subject_dir / 'left'}")
    left_img = load_image_gray(left_paths[0])
    left_tensor = torch.tensor(left_img, dtype=torch.float32).permute(2, 0, 1)  # 1,H,W

    # Iris right
    right_paths = get_bmps(subject_dir / "right")
    if not right_paths:
        raise FileNotFoundError(f"No right iris BMPs in {subject_dir / 'right'}")
    right_img = load_image_gray(right_paths[0])
    right_tensor = torch.tensor(right_img, dtype=torch.float32).permute(2, 0, 1)

    return {
        "fingerprint": fp_tensor.unsqueeze(0),  # (1,3,128,128)
        "iris_left": left_tensor.unsqueeze(0),  # (1,1,64,64)
        "iris_right": right_tensor.unsqueeze(0),  # (1,1,64,64)
    }


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None
    # Pick checkpoint with lowest loss (encoded in filename)
    return min(ckpts, key=lambda p: float(p.stem.split("loss_")[-1]))


def load_global_label_map(parquet_dir: Path) -> dict[int, int]:
    """Load subject_id → global_label mapping from any split parquet."""
    for split in ["train", "val", "test"]:
        p = parquet_dir / f"{split}.parquet"
        if p.exists():
            table = pq.read_table(p, columns=["subject_id", "label"])
            d = table.to_pydict()
            return {sid: lbl for sid, lbl in zip(d["subject_id"], d["label"], strict=True)}
    raise FileNotFoundError(f"No parquet files found in {parquet_dir}")


def confidence_bar(prob: float, width: int = 30) -> str:
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_banner() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}   MULTIMODAL BIOMETRIC RECOGNITION — INFERENCE DEMO{RESET}")
    print(f"{BOLD}{CYAN}   Fingerprint + Left Iris + Right Iris → Subject ID{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def run_demo(
    checkpoint_path: Path,
    dataset_root: Path,
    parquet_dir: Path,
    subject_id: int | None,
    top_k: int,
) -> None:
    print_banner()

    # ── Load label map ────────────────────────────────────────────────────
    label_map = load_global_label_map(parquet_dir)
    reverse_map = {v: k for k, v in label_map.items()}  # label → subject_id
    num_classes = max(label_map.values()) + 1
    all_subject_ids = sorted(label_map.keys())

    # ── Pick subject ──────────────────────────────────────────────────────
    # Use test-split subjects if possible
    test_p = parquet_dir / "test.parquet"
    if test_p.exists():
        t = pq.read_table(test_p, columns=["subject_id"])
        test_subjects = sorted(set(t["subject_id"].to_pylist()))
    else:
        test_subjects = all_subject_ids

    if subject_id is None:
        subject_id = random.choice(test_subjects)
        print(f"  Randomly selected subject from test set: {BOLD}Subject {subject_id}{RESET}")
    else:
        print(f"  Using specified subject: {BOLD}Subject {subject_id}{RESET}")

    subject_dir = dataset_root / str(subject_id)
    if not subject_dir.exists():
        print(f"{RED}ERROR: Subject directory not found: {subject_dir}{RESET}")
        return

    true_label = label_map[subject_id]
    print(f"  True global label : {BOLD}{true_label}{RESET}")
    print(f"  Fingerprint images: {len(get_bmps(subject_dir / 'Fingerprint'))}")
    print(f"  Left iris images  : {len(get_bmps(subject_dir / 'left'))}")
    print(f"  Right iris images : {len(get_bmps(subject_dir / 'right'))}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n{BOLD}Loading model...{RESET}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BiometricFusionModel(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"  Checkpoint  : {checkpoint_path.name}")
    print(f"  Epoch       : {epoch}")
    print(f"  Val loss    : {val_loss:.4f}")
    print(f"  Device      : {device}")
    print(f"  Num classes : {num_classes}")

    # ── Load images ───────────────────────────────────────────────────────
    print(f"\n{BOLD}Loading biometric images...{RESET}")
    inputs = load_subject_images(subject_dir)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"  Fingerprint tensor : {tuple(inputs['fingerprint'].shape)}")
    print(f"  Left iris tensor   : {tuple(inputs['iris_left'].shape)}")
    print(f"  Right iris tensor  : {tuple(inputs['iris_right'].shape)}")

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"\n{BOLD}Running inference...{RESET}")
    with torch.no_grad():
        logits = model(inputs)  # (1, num_classes)
        probs = functional.softmax(logits, dim=-1)[0]  # (num_classes,)

    k = min(top_k, num_classes)
    top_probs, top_labels = probs.topk(k)
    top_probs = top_probs.cpu().tolist()
    top_labels = top_labels.cpu().tolist()

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  PREDICTION RESULTS  (Top {k}){RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")
    print(f"  {'Rank':<5} {'Subject':>8} {'Label':>6}  {'Confidence':>10}   Bar")
    print(f"  {'─' * 55}")

    predicted_label = top_labels[0]
    predicted_subject = reverse_map.get(predicted_label, "?")
    correct = predicted_label == true_label

    for rank, (label, prob) in enumerate(zip(top_labels, top_probs, strict=True), 1):
        subj = reverse_map.get(label, "?")
        bar = confidence_bar(prob)
        pct = f"{prob * 100:.1f}%"

        if label == true_label:
            colour = GREEN
            marker = " ← TRUE"
        elif rank == 1:
            colour = RED
            marker = " ← PREDICTED"
        else:
            colour = RESET
            marker = ""

        print(
            f"  {colour}#{rank:<4} Subject {subj:>3}  [{label:>3}]  {pct:>8}   {bar}{marker}{RESET}"
        )

    print(f"\n{'─' * 60}")
    if correct:
        msg = f"  CORRECT  — Model identified Subject {subject_id} correctly!"
        print(f"{GREEN}{BOLD}  ✓{msg}{RESET}")
    else:
        msg = f"  INCORRECT — Predicted Subject {predicted_subject}, True Subject {subject_id}"
        print(f"{RED}{BOLD}  ✗{msg}{RESET}")

    # Top-5 check
    in_top5 = true_label in top_labels[:5]
    if in_top5 and not correct:
        print(f"{YELLOW}  ✓ Correct subject IS in top-5 predictions{RESET}")

    print(f"\n  Top-1 correct : {'Yes' if correct else 'No'}")
    print(f"  Top-5 correct : {'Yes' if in_top5 else 'No'}")
    print(
        f"  Confidence in true subject (label {true_label}): {probs[true_label].item() * 100:.2f}%"
    )
    print(f"{'─' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biometric inference demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: best in checkpoints/)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Subject ID to test (default: random from test set)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/data/IRIS and FINGERPRINT DATASET",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--parquet_dir", type=str, default="data/parquet", help="Path to parquet directory"
    )
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint(Path("checkpoints"))
        if ckpt_path is None:
            print(f"{RED}No checkpoint found in checkpoints/. Run training first.{RESET}")
            print("  python scripts/train.py")
            sys.exit(1)

    run_demo(
        checkpoint_path=ckpt_path,
        dataset_root=Path(args.data_dir),
        parquet_dir=Path(args.parquet_dir),
        subject_id=args.subject,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
