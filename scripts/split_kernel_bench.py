# scripts/split_kernel_bench.py
from datasets import load_dataset, concatenate_datasets
import random, json, pathlib
import pyarrow as pa
import numpy as np

# 1️⃣  load Level‑1 and Level‑2 splits (non‑streaming → indexable)
l1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1", streaming=False)
l2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2", streaming=False)

ds_full   = concatenate_datasets([l1, l2])
ds        = ds_full.select(range(200))

random.seed(42)
indices = list(range(len(ds)))
random.shuffle(indices)
holdout_idx = set(np.random.choice(len(ds), size=20, replace=False))

train, test = [], []

prompt_header = """You are an expert CUDA programmer. Your mission is to convert a given PyTorch operator into a high-performance CUDA kernel that is both correct and fast.

**Instructions:**
1.  Analyze the provided PyTorch code.
2.  Implement the equivalent logic in a CUDA kernel.
3.  Include a `main` function to initialize data, launch the kernel, and print the result to standard output. The output format must match the original PyTorch script's output exactly.
4.  Wrap your complete, runnable CUDA source code within `<code>` and `</code>` tags.

---

**Example:**

**PyTorch Operator:**
```python
import torch
import numpy as np

def vector_add(a, b):
    return a + b

# Initialization and execution
size = 128
a = torch.randn(size, dtype=torch.float32)
b = torch.randn(size, dtype=torch.float32)
c = vector_add(a, b)

# Print output for verification
for val in c:
    print(f"{val.item():.6f}", end=' ')
```

**Your CUDA Solution:**
<code>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Initialization
    int n = 128;
    size_t size = n * sizeof(float);
    float* h_a = new float[n];
    float* h_b = new float[n];
    // Using fixed values for reproducibility in the example
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    float* h_c = new float[n];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print output for verification
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        std::cout << h_c[i] << (i == n - 1 ? "" : " ");
    }
    std::cout << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
</code>

---

**Your Turn:**

Now, write a correct and fast CUDA kernel to replace the following PyTorch operator. Remember to include a `main` function and wrap the code in `<code>` tags.

**PyTorch Operator:**
"""

for i, ex in enumerate(ds):
    prompt_content = prompt_header + f"```python\n{ex['code']}\n```"

    record = {
        "task_id": ex.get("task_id", f"kb_{i:03d}"),
        "prompt": [{"role": "user", "content": prompt_content}],
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