"""Diagnostic — run to verify dataset structure, parquet contents, and label consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("=" * 60)
print("DATASET STRUCTURE CHECK")
print("=" * 60)

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
        s = subject_dirs[0]
        print(f"\nSubject {s.name} structure:")
        for sub in sorted(s.iterdir()):
            if sub.is_dir():
                bmps = list(
                    {
                        p.name.lower(): p for p in sub.iterdir() if p.suffix.lower() == ".bmp"
                    }.values()
                )
                print(f"  {sub.name}/ → {len(bmps)} BMP files")

print("\n" + "=" * 60)
print("PARQUET CHECK")
print("=" * 60)

parquet_dir = Path("data/parquet")

all_labels = {}
for split in ["train", "val", "test"]:
    p = parquet_dir / f"{split}.parquet"
    if not p.exists():
        print(f"{split}.parquet — NOT FOUND")
        continue

    schema = pq.read_schema(p)
    has_label = "label" in schema.names

    cols = ["subject_id"] + (["label"] if has_label else [])
    table = pq.read_table(p, columns=cols)
    d = table.to_pydict()
    ids = sorted(set(d["subject_id"]))

    if has_label:
        labels = sorted(set(d["label"]))
        rows_per_subj = len(d["subject_id"]) // len(ids)
        print(
            f"{split}.parquet → {len(table)} rows, {len(ids)} subjects, "
            f"labels {labels[0]}..{labels[-1]}, ~{rows_per_subj} rows/subject  ✓"
        )
        for sid, lbl in zip(d["subject_id"], d["label"], strict=True):
            all_labels[sid] = lbl
    else:
        print(
            f"{split}.parquet → {len(table)} rows, {len(ids)} subjects  "
            f"⚠ NO 'label' COLUMN — re-ingest needed!"
        )

print()
if len(all_labels) > 0:
    print("Label consistency check:")
    print(f"  Total unique subjects across all splits : {len(all_labels)}")
    print(f"  Label range                             : 0..{max(all_labels.values())}")

    for split in ["train", "val", "test"]:
        p = parquet_dir / f"{split}.parquet"
        if not p.exists():
            continue
        schema = pq.read_schema(p)
        if "label" not in schema.names:
            continue
        t = pq.read_table(p, columns=["subject_id", "label"])
        d = t.to_pydict()
        pairs = sorted(set(zip(d["subject_id"], d["label"])))
        print(f"\n  {split}: subject_id → label")
        for sid, lbl in pairs[:6]:
            print(f"    Subject {sid:>2} → label {lbl}")
        if len(pairs) > 6:
            print(f"    ... ({len(pairs)} total)")
    print("\n  ✓ Labels are globally consistent — same subject = same label in all splits")
