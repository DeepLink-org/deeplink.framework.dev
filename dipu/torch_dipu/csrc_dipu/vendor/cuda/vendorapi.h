// Copyright (c) 2023, DeepLink.

#pragma once
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define TRACK_FUN_CALL(TAG, x)                                       \
  {                                                                  \
    static bool enable = std::getenv("DIPU_TRACK_" #TAG) != nullptr; \
    if (enable) {                                                    \
      printf("[%d %s: %d]:%s\n", getpid(), __FILE__, __LINE__, x);   \
    }                                                                \
  }

#define DIPU_CALLCUDA(Expr)                                              \
  {                                                                      \
    TRACK_FUN_CALL(CUDA, #Expr);                                         \
    cudaError_t ret = Expr;                                              \
    TORCH_CHECK(ret == ::cudaSuccess, "call cuda error, expr = ", #Expr, \
                ", ret = ", ret);                                        \
  }

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

using diclComm_t = ncclComm_t;
using commUniqueId = ncclUniqueId;

}  // namespace dipu
