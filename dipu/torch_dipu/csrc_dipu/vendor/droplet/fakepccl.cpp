// #include <stdexcept>
// #include <string>
// #include <type_traits>
//
// #include <c10/core/ScalarType.h>
//
// #include "csrc_dipu/runtime/device/basedef.h"
// #include "csrc_dipu/runtime/devproxy/deviceproxy.h"
// #include <pccl.h>
// #endif  // USE_PCCL
// #include <torch/csrc/distributed/c10d/Types.hpp>
//
// #include <csrc_dipu/common.h>
// #include <csrc_dipu/runtime/device/diclapis.h>
//
// namespace dipu {
//
// namespace devapis {
// using diclCommValue_t = std::remove_pointer_t<diclComm_t>;
// constexpr diclCommValue_t kMagicComm = 0x5043434C;  // "PCCL"
//
// diclComm_t createDiclComm() { return new diclCommValue_t(kMagicComm); }
//
// void destroyDiclComm(diclComm_t comm) { delete comm; }
//
// void checkCommOrThrow(diclComm_t comm) {
//   if (comm == nullptr || *comm != kMagicComm) {
//     throw std::runtime_error("Invalid comm.");
//   }
// }
//
// [[noreturn]] void throwNotSupportedError() {
//   throw std::runtime_error(
//       "PCCL is not enabled. DIPU only allows single GPU communication.");
// }
//
// void checkNrankOrThrow(int nranks) {
//   if (nranks != 1) {
//     throwNotSupportedError();
//   }
// }
//
// void checkRankOrThrow(int rank) {
//   if (rank != 0) {
//     throwNotSupportedError();
//   }
// }
//
// void singleDeviceMemcpy(deviceStream_t stream, void* dst, const void* src,
//                         size_t nbytes) {
//   auto device = devproxy::current_device();
//   devproxy::memCopyD2DAsync(stream, nbytes, device, dst, device, src);
// }
//
// }  // namespace
//
// const int DICL_UNIQUE_ID_BYTES_SIZE = 0;
//
// DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
//   checkCommOrThrow(comm);
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
//                                        commUniqueId uniqueId, int rank,
//                                        int localDeviceId) {
//   checkNrankOrThrow(nranks);
//   checkRankOrThrow(rank);
//   DIPU_LOGW(
//       "PCCL is not enabled. DIPU will simulate single GPU "
//       "communication using memcpy.");
//   *comm = createDiclComm();
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
//   checkCommOrThrow(comm);
//   destroyDiclComm(comm);
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
//                                     size_t count, at::ScalarType datatype,
//                                     const ReduceOp& reduceOp, diclComm_t comm,
//                                     deviceStream_t stream) {
//   checkCommOrThrow(comm);
//   singleDeviceMemcpy(stream, recvbuff, sendbuff,
//                      count * at::elementSize(datatype));
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
//                                     size_t count, at::ScalarType datatype,
//                                     int root, diclComm_t comm,
//                                     deviceStream_t stream) {
//   checkCommOrThrow(comm);
//   singleDeviceMemcpy(stream, recvbuff, sendbuff,
//                      count * at::elementSize(datatype));
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
//                                     size_t count, at::ScalarType datatype,
//                                     diclComm_t comm, deviceStream_t stream) {
//   checkCommOrThrow(comm);
//   singleDeviceMemcpy(stream, recvBuf, sendBuf,
//                      count * at::elementSize(datatype));
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
//                                  size_t count, at::ScalarType datatype,
//                                  const ReduceOp& reduceOp, int root,
//                                  diclComm_t comm, deviceStream_t stream) {
//   checkCommOrThrow(comm);
//   checkRankOrThrow(root);
//   singleDeviceMemcpy(stream, recvbuff, sendbuff,
//                      count * at::elementSize(datatype));
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclReduceScatter(
//     void* sendBuf, void* recvBuf, size_t recvCount, at::ScalarType datatype,
//     const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
//   singleDeviceMemcpy(stream, recvBuf, sendBuf,
//                      recvCount * at::elementSize(datatype));
//   return DICL_SUCCESS;
// }
//
// DIPU_API diclResult_t diclSend(const void* sendbuff, size_t count,
//                                at::ScalarType datatype, int peer,
//                                diclComm_t comm, deviceStream_t stream) {
//   throwNotSupportedError();
//   return DICL_ERR_UNDEF;
// }
//
// DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count,
//                                at::ScalarType datatype, int peer,
//                                diclComm_t comm, deviceStream_t stream) {
//   throwNotSupportedError();
//   return DICL_ERR_UNDEF;
// }
//
// }  // end namespace devapis
//
// }  // end namespace dipu
