#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <cstring>
#include <cstdio>

int main()
{
    void *dst, *src;
    auto size = 32;
    cudaMalloc(&dst, size);
    cudaMallocHost(&src, size);
    unsigned long times = 256ul * 256 * 256 * 3;
    for (unsigned long i = 0; i < times; i++) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice);
    }
    return 0;
}