import pytest
from kevin_reward import compute_score

# Baseline correct PyTorch implementation for vector addition
# This will be used as the reference by the scoring function.
PYTORCH_ADD_VECTORS = """
import torch
import numpy as np

def vector_add(a, b):
    return a + b

size = 128
a = torch.ones(size, dtype=torch.float32) * 2
b = torch.ones(size, dtype=torch.float32) * 3
c = vector_add(a, b)

for val in c:
    print(f"{val.item()}", end=' ')
"""

# A correct CUDA implementation that should be faster
CUDA_CORRECT_FAST = """
<code>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 128;
    size_t size = n * sizeof(float);
    float h_a[128], h_b[128], h_c[128];
    for(int i=0; i<n; ++i) { h_a[i] = 2.0f; h_b[i] = 3.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add_kernel<<<1, 128>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        std::cout << h_c[i] << (i == n - 1 ? "" : " ");
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
</code>
"""

# Correct logic but implemented inefficiently to be slower
CUDA_CORRECT_SLOW = CUDA_CORRECT_FAST.replace(
    "int main() {",
    "int main() { for(volatile int i=0; i<50000000; ++i); "
)

# Compiles but produces incorrect output (adds a to itself)
CUDA_INCORRECT_OUTPUT = CUDA_CORRECT_FAST.replace("c[idx] = a[idx] + b[idx];", "c[idx] = a[idx] + a[idx];")

# This should cause a segmentation fault on the host
CUDA_RUNTIME_ERROR = CUDA_CORRECT_FAST.replace(
    "int main() {",
    "int main() { float* p = NULL; *p = 1.0f;"
)

# Code that will not compile
CUDA_COMPILE_ERROR = "<code> int main() { ERROR SYNTAX; } </code>"

# Not wrapped in code tags
NO_CODE_TAGS = "int main() { return 0; }"


def test_bad_format():
    """R0: Bad format -> 0.0"""
    assert compute_score(PYTORCH_ADD_VECTORS, NO_CODE_TAGS) == 0.0

def test_compile_error():
    """R1: Doesn't compile -> 0.1"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_COMPILE_ERROR) == pytest.approx(0.1)

def test_runtime_error():
    """R2: Runtime error -> 0.2"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_RUNTIME_ERROR) == pytest.approx(0.2)

def test_incorrect_output():
    """R3: Incorrect output -> 0.3"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_INCORRECT_OUTPUT) == pytest.approx(0.3)

def test_correct_but_slower():
    """R4: Correct but not faster -> 0.4"""
    # This test is sensitive to system load. It might be flaky.
    # It assumes the pytorch version is faster than the serially-launched kernel.
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_SLOW) == pytest.approx(0.4)

def test_correct_and_faster():
    """R5: Correct and faster -> > 0.4"""
    reward = compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_FAST)
    assert reward > 0.4 