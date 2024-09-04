#include "pccl.h"

pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId) {
    return 0;
}

pcclResult_t pcclCommInitRank(pcclComm_t* comm, int ndev, pcclUniqueId commId, int rank) {
    return 0;
}

pcclResult_t pcclCommInitAll(pcclComm_t* comms, int ndev, const int* devlist) {
    return 0;
}

pcclResult_t pcclCommDestroy(pcclComm_t comm) {
    return 0;
}

pcclResult_t pcclCommAbort(pcclComm_t comm) {
    return 0;
}

pcclResult_t pcclCommGetAsyncError(pcclComm_t comm, pcclResult_t *asyncError) {
    return 0;
}

const char* pcclGetErrorString(pcclResult_t result) {
    return nullptr;
}

const char* pcclGetLastError(pcclComm_t comm) {
    return nullptr;
}

pcclResult_t pcclCommCount(const pcclComm_t comm, int* count) {
    return 0;
}

pcclResult_t pcclCommCuDevice(const pcclComm_t comm, int* device) {
    return 0;
}

pcclResult_t pcclCommUserRank(const pcclComm_t comm, int* rank) {
    return 0;
}

pcclResult_t pcclGetVersion(int *version) {
    return 0;
}

pcclResult_t pcclReduce(const void* sendbuff, void* recvbuf, size_t count, pcclDataType_t datatype,
    pcclRedOp_t op, int root, pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclAllReduce(const void* sendbuff, void* recvbuff, size_t count, pcclDataType_t datatype,
    pcclRedOp_t op, pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, pcclDataType_t datatype, pcclRedOp_t op, pcclComm_t comm,
    tangStream_t stream) {
    return 0;
}

pcclResult_t pcclBroadcast(const void *sendbuff, void* recvbuff, size_t count, pcclDataType_t datatype, int root,
    pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclAllGather(const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclSend(const void* sendbuff, size_t count, pcclDataType_t datatype, int peer,
        pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclRecv(void* recvbuff, size_t count, pcclDataType_t datatype, int peer,
        pcclComm_t comm, tangStream_t stream) {
    return 0;
}

pcclResult_t pcclGroupStart(void) {
    return 0;
}

pcclResult_t pcclGroupEnd(void) {
    return 0;
}

