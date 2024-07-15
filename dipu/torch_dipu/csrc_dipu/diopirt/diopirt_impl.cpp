// Copyright (c) 2023, DeepLink.
#include "diopirt_impl.h"

#include <cstdio>
#include <mutex>
#include <string>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace diopihelper = dipu::diopi_helper;
using dipu::profile::RecordBlockCreator;

extern "C" {

DIOPI_RT_API const char* diopiGetVersion() {
  auto static const version =
      std::string("DIOPI Version: ") + std::to_string(DIOPI_VER_MAJOR) + "." +
      std::to_string(DIOPI_VER_MINOR) + "." + std::to_string(DIOPI_VER_PATCH);
  return version.c_str();
}

DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t pth,
                                             void** pptr) {
  *pptr = (reinterpret_cast<at::Tensor*>(pth))->data_ptr();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t pth,
                                                  const void** pptr) {
  *pptr = (reinterpret_cast<const at::Tensor*>(pth))->data_ptr();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t pth,
                                              diopiSize_t* size) {
  auto ptr = reinterpret_cast<const at::Tensor*>(pth);
  *size = diopiSize_t{ptr->sizes().data(), static_cast<int64_t>(ptr->dim())};
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t pth,
                                               diopiSize_t* stride) {
  auto ptr = reinterpret_cast<const at::Tensor*>(pth);
  *stride =
      diopiSize_t{ptr->strides().data(), static_cast<int64_t>(ptr->dim())};
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t pth,
                                              diopiDtype_t* dtype) {
  auto ptr = reinterpret_cast<const at::Tensor*>(pth);
  *dtype = diopihelper::toDiopiDtype(ptr->scalar_type());
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t pth,
                                               diopiDevice_t* device) {
  auto ptr = reinterpret_cast<const at::Tensor*>(pth);
  *device = (ptr->is_cpu() ? diopi_host : diopi_device);
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t pth,
                                              int64_t* numel) {
  if (pth == nullptr) {
    *numel = 0;
    return diopiSuccess;
  }

  auto ptr = reinterpret_cast<const at::Tensor*>(pth);
  *numel = ptr->numel();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th,
                                                 int64_t* itemsize) {
  auto ptr = reinterpret_cast<const at::Tensor*>(th);
  auto dtype = diopiDtype_t{};
  auto ret = diopiGetTensorDtype(th, &dtype);
  if (ret != diopiSuccess) {
    return ret;
  }

  *itemsize = diopihelper::getElemSize(dtype);
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStoragePtr(diopiConstTensorHandle_t pth,
                                                   void** pStoragePtr) {
  auto tensor = reinterpret_cast<const at::Tensor*>(pth);
  // Support both pt2.0 and pt2.1
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  *pStoragePtr = const_cast<void*>(tensor->storage().data());
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t
diopiGetTensorStorageOffset(diopiConstTensorHandle_t pth, int64_t* pOffset) {
  *pOffset = (reinterpret_cast<const at::Tensor*>(pth))->storage_offset();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t
diopiGetTensorStorageNbytes(diopiConstTensorHandle_t pth, size_t* pNbytes) {
  *pNbytes = (reinterpret_cast<const at::Tensor*>(pth))->storage().nbytes();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDeviceIndex(
    diopiConstTensorHandle_t pth, diopiDeviceIndex_t* pDevIndex) {
  *pDevIndex = (reinterpret_cast<const at::Tensor*>(pth))->device().index();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetCurrentDeviceIndex(
    diopiDeviceIndex_t* pDevIndex) {
  *pDevIndex = dipu::devproxy::current_device();
  return diopiSuccess;
}
DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx,
                                         diopiStreamHandle_t* stream) {
  *stream = ctx->stream;
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t* tensor,
                                             const diopiSize_t* size,
                                             const diopiSize_t* stride,
                                             const diopiDtype_t dtype,
                                             const diopiDevice_t device) {
  // TORCH_CHECK(tensor != nullptr && *tensor == nullptr, "invalid parameter
  // tensor");
  at::IntArrayRef at_dims(size->data, size->len);
  caffe2::TypeMeta at_type = diopihelper::toATenType(dtype);
  c10::DeviceType at_device = diopihelper::toATenDevice(device);
  auto options = at::TensorOptions(at_device).dtype(at_type);
  at::Tensor t;
  // Use nodispatch::empty to minimize dispatch operations when constructing on
  // a device.
  if (dipu::DIPU_DEVICE_TYPE == at_device) {
    if (stride) {
      at::IntArrayRef at_stride(stride->data, stride->len);
      t = dipu::native::nodispatch::empty_strided(at_dims, at_stride, options);
    } else {
      t = dipu::native::nodispatch::empty(at_dims, options);
    }
  } else {
    if (stride) {
      at::IntArrayRef at_stride(stride->data, stride->len);
      t = dipu::native::nodispatch::empty_strided_cpu(at_dims, at_stride,
                                                      options);
    } else {
      t = dipu::native::nodispatch::empty_cpu(at_dims, at_type.toScalarType(),
                                              at_device);
    }
  }

  ctx->arrays.emplace_back(std::move(t));
  *tensor = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t* tensor,
                                             int64_t num_bytes,
                                             diopiDevice_t device) {
  diopiSize_t size{&num_bytes, 1};
  return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8,
                            device);
}

DIOPI_RT_API diopiError_t diopiGeneratorGetState(diopiContextHandle_t ctx,
                                                 diopiConstGeneratorHandle_t th,
                                                 diopiTensorHandle_t* data) {
  auto generator = reinterpret_cast<const at::Generator*>(th);
  auto gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);

  at::Tensor tensor;
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    tensor = at::Tensor::wrap_tensor_impl(gen_impl->get_state());
  }

  ctx->arrays.emplace_back(std::move(tensor));
  *data = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorSetState(
    diopiGeneratorHandle_t th, diopiConstTensorHandle_t new_state) {
  auto generator = reinterpret_cast<at::Generator*>(th);
  auto gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
  auto ptr = reinterpret_cast<const at::Tensor*>(new_state);

  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    gen_impl->set_state(*(ptr->unsafeGetTensorImpl()));
  }

  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorGetSeedAndOffset(
    diopiGeneratorHandle_t th, uint64_t* seed, uint64_t* offset) {
  auto generator = reinterpret_cast<at::Generator*>(th);
  auto gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
  *offset = gen_impl->get_offset();
  *seed = gen_impl->current_seed();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorSetSeedAndOffset(
    diopiGeneratorHandle_t th, uint64_t seed, uint64_t offset) {
  auto generator = reinterpret_cast<at::Generator*>(th);
  auto gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
  gen_impl->set_offset(offset);
  gen_impl->set_current_seed(seed);
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordStart(const char* record_name,
                                           void** record) {
  *record = new RecordBlockCreator(record_name);
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordEnd(void** record) {
  TORCH_CHECK(record != nullptr, "invalid parameter record_function");
  auto dipu_record_block = static_cast<RecordBlockCreator*>(*record);
  dipu_record_block->end();
  delete dipu_record_block;
  *record = nullptr;
  return diopiSuccess;
}

}  // extern "C"
