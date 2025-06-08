# scripts/split_kernel_bench.py
from datasets import load_dataset, concatenate_datasets
import random, json, pathlib
import pyarrow as pa

# 1️⃣  load Level‑1 and Level‑2 splits (non‑streaming → indexable)
l1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1", streaming=False)
l2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2", streaming=False)

ds_full   = concatenate_datasets([l1, l2])
ds        = ds_full.select(range(200))

random.seed(42)
indices = list(range(len(ds)))
random.shuffle(indices)
holdout_idx  = set(indices[:20])

train, test = [], []
for i in range(len(ds)):
    ex = ds[i]
    
    # Create chat-formatted prompt for sglang multi-turn tool interactions
    prompt_content = (
        "Write a correct and fast CUDA kernel to replace the following "
        f"PyTorch operator:\n```python\n{ex['code']}\n```"
    )
    
    record = {
        "task_id": ex.get("task_id", f"kb_{i:03d}"),
        "prompt": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "meta": {
            "name": ex["name"],
            "level": ex["level"],
        },
    }
    (test if i in holdout_idx else train).append(record)

pathlib.Path("data").mkdir(exist_ok=True)
for name, blob in (
    ("kernelbench_train.parquet", train),
    ("kernelbench_holdout.parquet", test),
):
    # Convert list of dicts to PyArrow table
    table = pa.Table.from_pylist(blob)
    pa.parquet.write_table(table, f"data/{name}")
    print(f"Wrote data/{name} ({len(blob)} rows)")

print(f"✅  wrote {len(train)} train and {len(test)} hold-out tasks")