// cuda_shim.h -- Just enough of CUDA to type-check the GPU code on a
// machine that has no CUDA toolkit.
//
// The kernels are the part of this project that cannot be tested locally:
// there is no NVIDIA card on the build machine, so the .cu file is never
// compiled until it reaches a notebook.  That is a long time to find out
// about a typo.  This shim gives the compiler the builtins and the runtime
// functions the code uses, so that the file can be type-checked -- not run,
// just checked -- anywhere.
//
// The check lives in scripts/check_cuda_syntax.py.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

// The device code takes the CUDA branch of the portable core, which is what
// has to be checked.
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__

struct Dim3Shim {
    unsigned x = 1, y = 1, z = 1;
};
static Dim3Shim threadIdx, blockIdx, blockDim, gridDim;

inline void __syncwarp(unsigned = 0xffffffffu) {}
inline int  __any_sync(unsigned, int pred) { return pred; }
inline int  __all_sync(unsigned, int pred) { return pred; }
inline int  __shfl_up_sync(unsigned, int v, int delta) { (void)delta; return v; }
inline int  __shfl_down_sync(unsigned, int v, int delta) { (void)delta; return v; }
inline float __shfl_up_sync(unsigned, float v, int delta) { (void)delta; return v; }
inline float __shfl_down_sync(unsigned, float v, int delta) { (void)delta; return v; }
inline int  __shfl_sync(unsigned, int v, int) { return v; }
inline float __shfl_sync(unsigned, float v, int) { return v; }

typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 0,
    cudaMemcpyDeviceToHost = 1,
    cudaMemcpyDeviceToDevice = 2,
    cudaMemcpyHostToHost = 3
};

inline const char* cudaGetErrorString(cudaError_t) { return "no error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

template <class T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    *p = static_cast<T*>(std::malloc(n ? n : 1));
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n) std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* dst, int value, size_t n) {
    if (n) std::memset(dst, value, n);
    return cudaSuccess;
}
template <class T>
inline cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t n) {
    (void)symbol; (void)src; (void)n;
    return cudaSuccess;
}

typedef int cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = 0; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaEvent_t = 0) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    *ms = 0.f;
    return cudaSuccess;
}
