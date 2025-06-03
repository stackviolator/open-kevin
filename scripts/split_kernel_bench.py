from datasets import load_dataset, concatenate_datasets
import random, json, pathlib

# 1️⃣  load the two level splits (≈ 100 tasks each)
l1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
l2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2")

# 2️⃣  concatenate and keep only the first 200 rows
ds = concatenate_datasets([l1, l2])[:200]         # type: datasets.Dataset

# 3️⃣  reproducible train / hold‑out split
random.seed(42)
indices = list(range(len(ds)))
random.shuffle(indices)
holdout_idx = set(indices[:20])

train, test = [], []
for i, ex in enumerate(ds):
    record = {
        "task_id": ex["task_id"],
        "prompt": (
            "Write a correct and fast CUDA kernel to replace the following "
            f"PyTorch operator:\n```python\n{ex['pytorch_code']}\n```"
        ),
        "meta": {
            "shape_hint": ex["shape_hint"],
            "level": ex["level"],
            "timeout_ms": ex["timeout_ms"],
        },
    }
    (test if i in holdout_idx else train).append(record)

pathlib.Path("data").mkdir(exist_ok=True)
for name, blob in (
    ("kernelbench_train.jsonl", train),
    ("kernelbench_holdout.jsonl", test),
):
    with open(f"data/{name}", "w") as f:
        for r in blob:
            f.write(json.dumps(r) + "\n")

print("✅  wrote", len(train), "train and", len(test), "hold‑out tasks")
