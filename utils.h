#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__device__ __host__ auto proj(double3 p, int d) {
    return d == 0 ? double2 { p.y, p.z } :
           d == 1 ? double2 { p.z, p.x } :
                    double2 { p.x, p.y };
}
__device__ __host__ auto pick(double3 p, int d) {
    return d == 0 ? p.x :
           d == 1 ? p.y :
                    p.z;
}

__device__ __host__ inline auto fmin(double3 a, double3 b) {
    return double3 { ::fmin(a.x, b.x), ::fmin(a.y, b.y), ::fmin(a.z, b.z) };
}
__device__ __host__ inline auto fmin(double3 a, double3 b, double3 c) {
    return fmin(fmin(a, b), c);
}
__device__ __host__ inline auto fmax(double3 a, double3 b) {
    return double3 { ::fmax(a.x, b.x), ::fmax(a.y, b.y), ::fmax(a.z, b.z) };
}
__device__ __host__ inline auto fmax(double3 a, double3 b, double3 c) {
    return fmax(fmax(a, b), c);
}

// https://stackoverflow.com/a/27992604
#ifdef __INTELLISENSE__
struct dim3 {
    int x;
    int y;
    int z;
};
dim3 blockIdx;
dim3 blockDim;
dim3 threadIdx;
dim3 gridDim;
#define CU_DIM(grid, block)
#define CU_DIM_MEM(grid, block, bytes)
#define CU_DIM_MEM_STREAM(grid, block, bytes, stream)
extern int atomicAdd(int *, int);
extern size_t atomicAdd(size_t *, size_t);
#else
#define CU_DIM(grid, block) <<<(grid), (block)>>>
#define CU_DIM_MEM(grid, block, bytes) <<<(grid), (block), (bytes)>>>
#define CU_DIM_MEM_STREAM(grid, block, bytes, stream) <<<(grid), (block), (bytes), (stream)>>>
#endif

#define cuIdx(D) (threadIdx.D + blockIdx.D * blockDim.D)
#define cuDim(D) (blockDim.D * gridDim.D)

#define CALL_AND_ASSERT(call, success, message) do { \
    auto code = call;       \
    if (code != success) { \
        fprintf(stderr, "call %s failed with message \"%s\" at %s:%d\n", #call, message(code), __FILE__, __LINE__); \
        exit(-1);           \
    }                       \
} while (0)

#define  CUDA_ASSERT(call) CALL_AND_ASSERT(call, cudaSuccess, cudaGetErrorString)

template <typename T>
struct buffer {
    T *ptr = nullptr;
    size_t len = 0;
    __device__ __host__ __forceinline__
    auto &operator[](int i) {
        return ptr[i];
    }
};

template <typename T>
struct device_vector : public buffer<T> {
    device_vector(size_t len = 0) {
        if (ptr) {
            CUDA_ASSERT(cudaFree(ptr));
        }
        if (len) {
            CUDA_ASSERT(cudaMalloc(&ptr, len));
        }
        this->len = len;
    }
    ~device_vector() {
        if (ptr) {
            CUDA_ASSERT(cudaFree(ptr));
        }
    }
};

template <typename T>
auto buffer_from(std::vector<T> &vec) {
    return buffer<T> { vec.data(), vec.size() };
};

#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;
auto clock_now() {
    return high_resolution_clock::now();
}
auto seconds_since(steady_clock::time_point &start) {
    std::chrono::duration<double> duration = clock_now() - start;
    return duration.count();
}
