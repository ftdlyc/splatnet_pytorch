#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK_ERROR(err) gpuAssert(err, __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define CUDA_CHECK_KERNEL_ERROR() cudaDeviceSynchronize(); gpuAssert(cudaGetLastError(), __FILE__, __LINE__)

inline const char *cublasGetErrorString(cublasStatus_t stat) {
  switch (stat) {
    case CUBLAS_STATUS_NOT_INITIALIZED:return "The cuBLAS library was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "Resource allocation failed inside the cuBLAS library";
    case CUBLAS_STATUS_INVALID_VALUE: return "An unsupported value or parameter was passed to the function ";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "The function requires a feature absent from the device architecture";
    case CUBLAS_STATUS_MAPPING_ERROR: return "An access to GPU memory space failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "The GPU program failed to execute";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "An internal cuBLAS operation failed";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "The functionnality requested is not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "The functionnality requested requires some license";
  }
  return "Unknown error";
};

#define CUBLAS_CHECK_ERROR(err) cublasAssert(err, __FILE__, __LINE__)
inline void cublasAssert(cublasStatus_t stat, const char *file, int line, bool abort = true) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLASassert: %s %s %d\n", cublasGetErrorString(stat), file, line);
    if (abort) exit(stat);
  }
}

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_ASSERT(!(x.type().is_cuda()), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_TYPE(x, y) AT_ASSERT(x.type().scalarType() == y, #x " must be " #y)

#endif //CUDA_UTILS_H
