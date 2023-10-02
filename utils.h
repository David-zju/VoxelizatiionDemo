#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __host__ auto rotate(T p, int d) {
    return d == 0 ? T { p.y, p.z, p.x } :
           d == 1 ? T { p.z, p.x, p.y } :
                    T { p.x, p.y, p.z };
}

template <typename T>
__device__ __host__ auto revert(T p, int d) {
    return d == 0 ? T { p.z, p.x, p.y } :
           d == 1 ? T { p.y, p.z, p.x } :
                    T { p.x, p.y, p.z };
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

template <typename T>
struct buffer {
    T *ptr = nullptr;
    size_t len = 0;
    __device__ __host__ __forceinline__
    auto &operator[](int i) {
        return ptr[i];
    }
};

#include <vector>
using std::vector;

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
