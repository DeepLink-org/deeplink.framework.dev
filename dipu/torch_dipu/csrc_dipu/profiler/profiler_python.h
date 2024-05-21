#pragma once

#include <memory>

#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace dipu {
namespace profile {

class DIPURecordQueue;
inline std::unique_ptr<torch::profiler::impl::python_tracer::PythonTracerBase>
makeTracer(DIPURecordQueue* queue) { return {}; }

inline void init() {}

}  // namespace profile
}  // namespace dipu
