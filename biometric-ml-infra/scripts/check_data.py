"""Diagnostic — run to verify dataset structure and parquet contents."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("=" * 60)
print("DATASET STRUCTURE CHECK")
print("=" * 60)

# Check raw data folder
raw = Path("data/data")
dataset_root = raw / "IRIS and FINGERPRINT DATASET"
if not dataset_root.exists():
    dataset_root = raw
    print(f"Looking in: {raw}")
else:
    print(f"Found dataset at: {dataset_root}")

if dataset_root.exists():
    subject_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit())
    print(f"Subjects found: {len(subject_dirs)}")
    if subject_dirs:
        # Show first subject structure
        s = subject_dirs[0]
        print(f"\nSubject {s.name} structure:")
        for sub in sorted(s.iterdir()):
            if sub.is_dir():
                bmps = list(sub.glob("*.bmp")) + list(sub.glob("*.BMP"))
                print(f"  {sub.name}/ → {len(bmps)} BMP files")
else:
    print(f"ERROR: Dataset not found at {dataset_root}")
    print("Contents of data/data/:")
    if raw.exists():
        for item in raw.iterdir():
            print(f"  {item.name}")
    else:
        print("  data/data/ does not exist!")

print("\n" + "=" * 60)
print("PARQUET CHECK")
print("=" * 60)

import pyarrow.parquet as pq
parquet_dir = Path("data/parquet")
for split in ["train", "val", "test"]:
    p = parquet_dir / f"{split}.parquet"
    if not p.exists():
        print(f"{split}.parquet — NOT FOUND")
        continue
    table = pq.read_table(p, columns=["subject_id"])
    ids = sorted(set(table["subject_id"].to_pylist()))
    print(f"{split}.parquet → {len(table)} rows, {len(ids)} subjects: {ids}")
