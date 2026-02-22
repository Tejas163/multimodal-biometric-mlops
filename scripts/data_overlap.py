import pyarrow.parquet as pq

def check_labels(split):
    table = pq.read_table(f"data/parquet/{split}.parquet", columns=["subject_id", "label"])
    d = table.to_pydict()
    labels = set(d["label"])
    return labels

train_labels = check_labels("train")
val_labels = check_labels("val")
test_labels = check_labels("test")

print(f"Train labels: {len(train_labels)} labels, {sorted(train_labels)[:20]}...")
print(f"Val labels:   {len(val_labels)} labels, {sorted(val_labels)}")
print(f"Test labels:  {len(test_labels)} labels, {sorted(test_labels)}")

print(f"\nVal in train?  {val_labels.issubset(train_labels)}")
print(f"Test in train? {test_labels.issubset(train_labels)}")
print(f"Missing from train: {sorted(val_labels - train_labels)}")