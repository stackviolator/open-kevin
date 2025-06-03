# scripts/split_kernel_bench.py
from datasets import load_dataset, concatenate_datasets
import random, json, pathlib

# 1️⃣  load Level‑1 and Level‑2 splits (non‑streaming → indexable)
l1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1", streaming=False)
l2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2", streaming=False)

# 2️⃣  concat then take the first 200 examples while REMAINING a Dataset
ds_full   = concatenate_datasets([l1, l2])
ds        = ds_full.select(range(200))          # ← keeps Dataset semantics

# 3️⃣  reproducible random 180 / 20 split
random.seed(42)
indices       = list(range(len(ds)))            # 0 … 199
random.shuffle(indices)
holdout_idx   = set(indices[:20])

train, test = [], []
for i in range(len(ds)):
    ex = ds[i]
    record = {
        "task_id": ex.get("task_id", f"kb_{i:03d}"),
        "prompt": (
            "Write a correct and fast CUDA kernel to replace the following "
            f"PyTorch operator:\n```python\n{ex['code']}\n```"
        ),
        "meta": {
            "name": ex["name"],
            "level": ex["level"],
        },
    }
    (test if i in holdout_idx else train).append(record)

# 4️⃣  write JSONL files
pathlib.Path("data").mkdir(exist_ok=True)
for name, blob in (
    ("kernelbench_train.jsonl", train),
    ("kernelbench_holdout.jsonl", test),
):
    with open(f"data/{name}", "w") as f:
        for r in blob:
            f.write(json.dumps(r) + "\n")

print(f"✅  wrote {len(train)} train and {len(test)} hold‑out tasks")

