from datasets import load_dataset, concatenate_datasets
import random, pathlib
import pyarrow as pa
import numpy as np

"""Split KernelBench Level‑1 and Level‑2 into a small train/hold‑out set
with an updated prompt that mirrors exactly what `test_reward.py` now
expects: the model must return a single **Python script** whose body uses
`torch.utils.cpp_extension.load_inline` to JIT‑compile an inline CUDA
kernel and expose it via a `torch.nn.Module` subclass. The output is
wrapped **solely** in `<code> … </code>` tags with *no* extra prose.
"""

# 1️⃣  Load Level‑1 and Level‑2 splits (non‑streaming → indexable)
l1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1", streaming=False)
l2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2", streaming=False)

ds_full = concatenate_datasets([l1, l2])
# Keep a small subset for the demo artefact
_ds = ds_full.select(range(200))

# ---------------------------------------------------------------------------
# 🔀 Train / hold‑out split (20 examples for hold‑out)
# ---------------------------------------------------------------------------

random.seed(42)
indices = list(range(len(_ds)))
random.shuffle(indices)
_holdout = set(np.random.choice(len(_ds), size=20, replace=False))

train, holdout = [], []

for idx, ex in enumerate(_ds):
    prompt_content = ex['code']

    record = {
        "task_id": ex.get("task_id", f"kb_{idx:03d}"),
        "question": prompt_content,
        "answer": "",
        "meta": {
            "name": ex["name"],
            "level": ex["level"],
        },
    }
    (holdout if idx in _holdout else train).append(record)

# ---------------------------------------------------------------------------
# 💾  Persist to Parquet
# ---------------------------------------------------------------------------

pathlib.Path("data").mkdir(exist_ok=True)
for fname, blob in (
    ("kernelbench_train.parquet", train),
    ("kernelbench_holdout.parquet", holdout),
):
    pa.parquet.write_table(pa.Table.from_pylist(blob), f"data/{fname}")
    print(f"Wrote {fname} ({len(blob)} rows)")

print(f"✅  wrote {len(train)} train and {len(holdout)} hold‑out tasks")
