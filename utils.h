#pragma once
#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <chrono>

#include <cuda_runtime.h>

// https://stackoverflow.com/a/27992604
#ifdef __INTELLISENSE__
struct dim3 {
    int x;
    int y;
    int z;
    dim3(int, int, int);
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

namespace bocchi {

using namespace std;

template <typename T>
__device__ __host__ __forceinline__ auto rotate(T p, int d) {
    return d == 0 ? T { p.y, p.z, p.x } :
           d == 1 ? T { p.z, p.x, p.y } :
                    p;
}

template <typename T>
__device__ __host__ __forceinline__ auto revert(T p, int d) {
    return d == 0 ? T { p.z, p.x, p.y } :
           d == 1 ? T { p.y, p.z, p.x } :
                    p;
}

template <typename T>
__device__ __host__ __forceinline__ auto fmin3(T a, T b) {
    return T { fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z) };
}
__device__ __host__ __forceinline__ auto fmin3(double3 a, double3 b, double3 c) {
    return fmin3(fmin3(a, b), c);
}
template <typename T>
__device__ __host__ __forceinline__ auto fmax3(T a, T b) {
    return T { fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z) };
}
__device__ __host__ __forceinline__ auto fmax3(double3 a, double3 b, double3 c) {
    return fmax3(fmax3(a, b), c);
}

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
__global__ void kernel_reset(buffer<T> arr, T val) {
    for (int i = cuIdx(x); i < arr.len; i += cuDim(x)) {
        arr[i] = val;
    }
}

template <typename T>
struct device_vector : public buffer<T> {
    device_vector(size_t capacity = 0) {
        ptr = nullptr;
        if (capacity) {
            resize(capacity);
        }
    }
    device_vector(vector<T> &vec) {
        ptr = nullptr;
        auto capacity = vec.size();
        if (capacity) {
            resize(capacity);
            CUDA_ASSERT(cudaMemcpy(ptr, vec.data(), capacity * sizeof(T), cudaMemcpyDefault));
        }
    }
    auto resize(size_t capacity) {
        if (capacity > len) {
            if (ptr) {
                CUDA_ASSERT(cudaFree(ptr));
            }
            CUDA_ASSERT(cudaMalloc(&ptr, capacity * sizeof(T)));
        }
        len = capacity;
    }
    auto copy() {
        vector<T> vec(len);
        return copy_to(vec);
    }
    auto &copy_to(vector<T> &vec) {
        vec.resize(len);
        CUDA_ASSERT(cudaMemcpy(vec.data(), ptr, len * sizeof(T), cudaMemcpyDefault));
        return vec;
    }
    auto &copy_from(vector<T> &vec) {
        auto len = vec.size();
        resize(len);
        CUDA_ASSERT(cudaMemcpy(ptr, vec.data(), len * sizeof(T), cudaMemcpyDefault));
        return vec;
    }
    ~device_vector() {
        if (ptr) {
            CUDA_ASSERT(cudaFree(ptr));
            ptr = nullptr;
        }
        len = 0;
    }
};

auto clock_now() {
    return chrono::high_resolution_clock::now();
}
auto seconds_since(chrono::steady_clock::time_point &start) {
    chrono::duration<double> duration = clock_now() - start;
    return duration.count();
}

};
