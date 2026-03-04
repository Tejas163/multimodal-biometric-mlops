"""
MLflow inference server script.

Usage:
    # Start MLflow UI (local)
    python scripts/serve_mlflow.py

    # Start MLflow with model serving
    python scripts/serve_mlflow.py --serve-model --port 5001

    # Run inference demo
    python scripts/serve_mlflow.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.inference.pipeline import InferencePipeline
from biometric_ml.utils.logging import setup_logging


def start_mlflow_ui(tracking_uri: str, port: int):
    """Launch MLflow UI."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    print()
    print("=" * 60)
    print("Starting MLflow UI")
    print(f"Tracking URI: {tracking_uri}")
    print(f"UI Port: {port}")
    print("=" * 60)
    print()
    print(f"Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop")
    print()

    import subprocess

    subprocess.run(["mlflow", "ui", "--port", str(port), "--backend-store-uri", tracking_uri])


def serve_model(
    tracking_uri: str,
    model_name: str,
    port: int,
    device: str,
):
    """Serve model via MLflow's built-in REST API."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    print()
    print("=" * 60)
    print("Starting MLflow model serving")
    print(f"Model: {model_name}")
    print(f"Port: {port}")
    print("=" * 60)
    print()

    import subprocess

    subprocess.run(
        [
            "mlflow",
            "models",
            "serve",
            "-m",
            f"models:/{model_name}/Production",
            "-p",
            str(port),
            "--env-manager",
            "local",
        ]
    )


def run_inference_demo(tracking_uri: str, model_name: str):
    """Run a quick inference demo."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading model from MLflow registry...")

    pipeline = InferencePipeline.from_registry(
        model_name=model_name,
        stage="Production",
        tracking_uri=tracking_uri,
    )

    logger.info("Running inference demo...")

    sample = {
        "fingerprint": torch.randn(3, 128, 128).tolist(),
        "iris_left": torch.randn(1, 64, 64).tolist(),
        "iris_right": torch.randn(1, 64, 64).tolist(),
    }

    result = pipeline.predict(sample)

    print()
    print("=" * 60)
    print("Inference Result")
    print("=" * 60)
    print("Top-5 predictions:")
    for i, (cls, prob) in enumerate(zip(result.top_k_ids, result.top_k_probs, strict=True), 1):
        print(f"  {i}. Class {cls}: {prob:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="MLflow inference server")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument(
        "--model-name", default="BiometricFusionModel", help="Model name in registry"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")
    parser.add_argument("--serve-model", action="store_true", help="Serve model via REST API")
    parser.add_argument("--demo", action="store_true", help="Run inference demo")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    args = parser.parse_args()

    setup_logging(level="INFO", json_format=False)

    if args.demo:
        run_inference_demo(args.tracking_uri, args.model_name)
    elif args.serve_model:
        serve_model(args.tracking_uri, args.model_name, args.port, args.device)
    else:
        start_mlflow_ui(args.tracking_uri, args.port)


if __name__ == "__main__":
    main()
