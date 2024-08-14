// Copyright (c) 2023, DeepLink.
#pragma once

#include <iostream>
#include <mutex>
#include <thread>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/core/MemChecker.h"
#include "csrc_dipu/runtime/device/deviceapis.h"

namespace dipu {

// TODO(allocator): refactor it someday.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DIPU_DEBUG_ALLOCATOR(mask, x)                                          \
  {                                                                            \
    static int value = []() {                                                  \
      auto env = std::getenv("DIPU_DEBUG_ALLOCATOR");                          \
      return env ? std::atoi(env) : 0;                                         \
    }();                                                                       \
    if ((mask & value) == mask) {                                              \
      std::cout << "[" << std::this_thread::get_id() << "]" << x << std::endl; \
    }                                                                          \
  }
// NOLINTEND(bugprone-macro-parentheses)

class DIPU_API DIPURawDeviceAllocator : public c10::Allocator {
 public:
  DIPURawDeviceAllocator();

#if DIPU_TORCH_VERSION >= 20300
  c10::DataPtr allocate(size_t size) override;
#else
  c10::DataPtr allocate(size_t size) const override;
#endif

  c10::DeleterFnPtr raw_deleter() const override;

#if DIPU_TORCH_VERSION >= 20300
  void copy_data(void* dest, const void* src,
                 std::size_t count) const override {
    dipu::devapis::memCopyD2D(count, 0, dest, 0, src);
  }
#endif

 private:
  // TODO(allocator): refactor it someday.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::mutex mutex_;
  c10::DataPtr allocate(size_t nbytes, c10::DeviceIndex device_index) const;
};

class DIPURawHostAllocator : public c10::Allocator {
 public:
#if DIPU_TORCH_VERSION >= 20300
  c10::DataPtr allocate(size_t size) override;
#else
  c10::DataPtr allocate(size_t size) const override;
#endif

  c10::DeleterFnPtr raw_deleter() const override;

#if DIPU_TORCH_VERSION >= 20300
  void copy_data(void* dest, const void* src,
                 std::size_t count) const override {
    dipu::devapis::memCopyD2D(count, 0, dest, 0, src);
  }
#endif
};

DIPU_API bool isPinnedPtr(const void* ptr);

}  // namespace dipu
