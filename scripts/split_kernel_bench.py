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
# 📝 Prompt shown to the model during fine‑tuning / inference
# ---------------------------------------------------------------------------
# ⚠️  The guidance below is *precisely* aligned with `compute_score` in
#     `test_reward.py`. Do **not** change lightly.
# ---------------------------------------------------------------------------

prompt_header = """You are an expert CUDA programmer. Your mission is to convert a given PyTorch operator into a **high‑performance** custom CUDA kernel *wrapped* inside a Python script via **`torch.utils.cpp_extension.load_inline`**.

**Instructions — your output must:**
1. Analyse the provided PyTorch code.
2. Implement the equivalent logic in a CUDA kernel (C++/CUDA).
3. In a *single* Python script:
   • JIT‑compile the kernel using `load_inline` (supplying both `cpp_sources` and `cuda_sources`).
   • Expose the kernel via a subclass of `torch.nn.Module` whose `forward` method calls the compiled function.
4. Output **only one** `<code> … </code>` block containing the full, runnable Python script — **no text** before or after the tags.
5. Ensure your implementation is **correct** and noticeably **faster** than the baseline PyTorch operator.

---

**Example (for vector addition):**

**PyTorch Operator:**
```python
import torch

def vector_add(a, b):
    return a + b

size = 128
a = torch.randn(size)
b = torch.randn(size)
print(vector_add(a, b))
```

**Your CUDA Solution:**
<code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ↳ Inline CUDA kernel & signature
cuda_source = r'''
#include <torch/extension.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    const int n = a.numel();
    auto c = torch::empty_like(a);
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
'''
cpp_source = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

add_mod = load_inline(
    name="vector_add",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def forward(self, a, b):
        return add_mod.add_cuda(a, b)
</code>

---

**Your Turn:**

Following the same pattern, write a **correct** and **fast** inline‑CUDA solution for the operator below.

**PyTorch Operator:**"""

# ---------------------------------------------------------------------------
# 🔀 Train / hold‑out split (20 examples for hold‑out)
# ---------------------------------------------------------------------------

random.seed(42)
indices = list(range(len(_ds)))
random.shuffle(indices)
_holdout = set(np.random.choice(len(_ds), size=20, replace=False))

train, holdout = [], []

for idx, ex in enumerate(_ds):
    prompt_content = prompt_header + f"```python\n{ex['code']}\n```"

    record = {
        "task_id": ex.get("task_id", f"kb_{idx:03d}"),
        "prompt": [{"role": "user", "content": prompt_content}],
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
