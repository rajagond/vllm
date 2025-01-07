#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define CUDA_CHECK_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t result = call;                                                 \
    if (result != cudaSuccess) {                                               \
      std::cout << "CUDA error: " << cudaGetErrorString(result) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(result);                                                            \
    }                                                                          \
  } while (0)


void copy_2d(
  torch::Tensor& dest, 
  const torch::Tensor& src,
  int64_t N, 
  int64_t start_idx, int64_t num_elems) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    dest.scalar_type(),
    "copy_2d",
    [&] {
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dest.data_ptr<scalar_t>() + start_idx * N, src.data_ptr<scalar_t>() 
            + start_idx * N, num_elems * N  * sizeof(scalar_t), cudaMemcpyDeviceToDevice, stream));
    });
}