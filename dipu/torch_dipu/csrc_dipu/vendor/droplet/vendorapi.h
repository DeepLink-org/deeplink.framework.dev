#pragma once
#include <cstdio>

#include <c10/util/Exception.h>
#ifdef USE_PCCL
#include "pccl.h"
#endif  // USE_PCCL
#include <tang_runtime.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define DIPU_CALLDROPLET(Expr)                                              \
  {                                                                         \
    tangError_t ret = Expr;                                                 \
    if (ret != tangSuccess) {                                               \
      printf("call a tangrt function (%s) failed. return code=%d\n", #Expr, \
             ret);                                                          \
      fflush(stdout);                                                       \
      throw std::runtime_error("dipu device error");                        \
    }                                                                       \
  }

using deviceStream_t = tangStream_t;
#define deviceDefaultStreamLiteral nullptr
using deviceEvent_t = tangEvent_t;
using deviceHandle_t = tangContext_t*;
#ifdef USE_PCCL
using diclComm_t = pcclComm_t;
using commUniqueId = pcclUniqueId;
#else   // USE_PCCL
using diclComm_t = uint32_t*;
struct commUniqueId {};
#endif  // USE_PCCL

}  // namespace dipu
