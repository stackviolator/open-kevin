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

prompt_header = """You are given the following PyTorch architecture:

"""

prompt_footer = """

Replace pytorch operators in the given architecture with raw CUDA kernels, 
optimizing for performance on NVIDIA A100 (e.g. shared memory, kernel fusion, 
warp primitives, vectorization,...). 

Use torch.utils.cpp_extension.load_inline and name your optimized output 
architecture ModelNew. 

You're not allowed to use torch.nn (except for Parameter, containers, and init). 
The input and output have to be on CUDA device. 

Your answer must be the complete new architecture (no testing code, no other code): 
it will be evaluated and you will be given feedback on its correctness and speedup 
so you can keep iterating, trying to maximize the speedup.

**Important: Wrap your entire code response in <code> </code> tags. Do not include any other text or explanations outside these tags.**

Here's an example:

<code>
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
\"\"\"

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
</code>

After providing your complete ModelNew architecture, you must call calc_cuda_reward(cuda_src=<your_generated_code>) with your complete Python script as the parameter. Summarize your changes in a few sentences after the tool call.

**Your Turn:**

Optimize the following PyTorch architecture:"""

# ---------------------------------------------------------------------------
# 🔀 Train / hold‑out split (20 examples for hold‑out)
# ---------------------------------------------------------------------------

random.seed(42)
indices = list(range(len(_ds)))
random.shuffle(indices)
_holdout = set(np.random.choice(len(_ds), size=20, replace=False))

train, holdout = [], []

for idx, ex in enumerate(_ds):
    prompt_content = prompt_header + f"<original_code>\n{ex['code']}\n</original_code>" + prompt_footer

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
