from setuptools import setup, find_packages

setup(
    name="biometric-ml-infra",
    version="1.0.0",
    description="Production ML infrastructure for multimodal biometric user recognition",
    author="Tejas163",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",  # needed even if not used — transitive dep of mlflow
        "mlflow>=2.10.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyarrow>=14.0.0",
        "numpy>=1.24.0,<2.0.0",
        "Pillow>=10.0.0",
        "ray>=2.43.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
            "types-PyYAML",
        ],
    },
    entry_points={
        "console_scripts": [
            "biometric-ingest=scripts.ingest_data:main",
            "biometric-train=scripts.train:main",
        ],
    },
)
