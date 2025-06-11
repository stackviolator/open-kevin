system_prompt = '''Replace pytorch operators in the given architecture with raw CUDA kernels, 
optimizing for performance on NVIDIA A100 (e.g. shared memory, kernel fusion, 
warp primitives, vectorization,...). 

Use torch.utils.cpp_extension.load_inline and name your optimized output 
architecture ModelNew. 

You're not allowed to use torch.nn (except for Parameter, containers, and init). 
The input and output have to be on CUDA device. 

Your answer must be the complete new architecture (no testing code, no other code): 
it will be evaluated and you will be given feedback on its correctness and speedup 
so you can keep iterating, trying to maximize the speedup.

Here's an example:

<code>
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
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
"""

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

Before you start, adhere to the following tag guidelines:

<think>
Provide your complete, step-by-step reasoning here. Explain which PyTorch operators you will replace, how you design and fuse CUDA kernels, your memory-hierarchy considerations, warp-level optimizations, and any other design choices. Do NOT include any code inside this block.
</think>

<code>
Provide ONLY the final, executable Python code that defines `ModelNew`, including any required inline C++/CUDA sources. This block will be extracted and run by the evaluator, so it must be fully self-contained. Do NOT repeat your reasoning here and do not include anything that is not valid Python/C++/CUDA code.
</code>

Do not output anything outside the <think> and <code> blocks.

''' 