import pytest
from kevin_reward import compute_score

# Baseline correct PyTorch implementation for vector addition
# This will be used as the reference by the scoring function.
PYTORCH_ADD_VECTORS = """
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, a, b):
        return a + b

def get_inputs():
    a = torch.randn(10000, dtype=torch.float32)
    b = torch.randn(10000, dtype=torch.float32)
    return [a, b]

def get_init_inputs():
    return []
"""

# A correct CUDA implementation that should be faster
CUDA_CORRECT_FAST = """
<code>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <metadata_file> <output_file>" << std::endl;
        return 1;
    }
    
    std::string metadata_file = argv[1];
    std::string output_file = argv[2];
    
    // Simple approach: assume input_0.bin and input_1.bin exist in same directory
    std::string base_dir = metadata_file.substr(0, metadata_file.find_last_of("/\\") + 1);
    std::string a_file = base_dir + "input_0.bin";
    std::string b_file = base_dir + "input_1.bin";
    
    // Get file size to determine number of elements
    std::ifstream fa(a_file, std::ios::binary | std::ios::ate);
    size_t file_size = fa.tellg();
    fa.seekg(0);
    int n = file_size / sizeof(float);
    
    size_t size = n * sizeof(float);
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    // Read input data from files
    std::ifstream fb(b_file, std::ios::binary);
    fa.read(reinterpret_cast<char*>(h_a), size);
    fb.read(reinterpret_cast<char*>(h_b), size);
    fa.close();
    fb.close();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Write output to file
    std::ofstream fout(output_file, std::ios::binary);
    fout.write(reinterpret_cast<char*>(h_c), size);
    fout.close();
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
</code>
"""

# Correct logic but implemented inefficiently to be slower
CUDA_CORRECT_SLOW = CUDA_CORRECT_FAST.replace(
    "int main(int argc, char* argv[]) {",
    "int main(int argc, char* argv[]) { for(volatile int i=0; i<100000000; ++i); "
)

# Compiles but produces incorrect output (adds a to itself)
CUDA_INCORRECT_OUTPUT = CUDA_CORRECT_FAST.replace("c[idx] = a[idx] + b[idx];", "c[idx] = a[idx] + a[idx];")

# This should cause a segmentation fault on the host
CUDA_RUNTIME_ERROR = CUDA_CORRECT_FAST.replace(
    "int main(int argc, char* argv[]) {",
    "int main(int argc, char* argv[]) { float* p = NULL; *p = 1.0f;"
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