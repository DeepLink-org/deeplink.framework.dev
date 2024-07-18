/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiTensorHandle_t transposeTensor(diopiContextHandle_t ctx, diopiConstTensorHandle_t input) {
    AscendTensor inputTensor(input);
    diopiTensorHandle_t inputTrans;

    std::vector<int64_t> inputTransShape(inputTensor.shape().begin(), inputTensor.shape().end());
    inputTransShape[inputTensor.dim() - 1] = inputTensor.shape(inputTensor.dim() - 2);
    inputTransShape[inputTensor.dim() - 2] = inputTensor.shape(inputTensor.dim() - 1);
    diopiSize_t inputTransSize{inputTransShape.data(), inputTransShape.size()};

    std::vector<int64_t> dims(inputTensor.dim());
    std::iota(dims.begin(), dims.end(), 0);
    dims[inputTensor.dim() - 1] = inputTensor.dim() - 2;
    dims[inputTensor.dim() - 2] = inputTensor.dim() - 1;
    diopiSize_t permuteDims = vectorToDiopiSize(dims);

    diopiRequireTensor(ctx, &inputTrans, &inputTransSize, nullptr, inputTensor.dtype(), diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, input, permuteDims, inputTrans);
    return inputTrans;
}

DIOPI_API diopiError_t diopiGroupedGemm(diopiContextHandle_t ctx, diopiTensorHandle_t outs, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t weights,
                                        diopiConstTensorHandle_t batchSizes, bool transInputs, bool transWeights) {
    // calculate groupList
    diopiTensorHandle_t groupListTensor;
    diopiSize_t batchSizesShape;
    diopiGetTensorShape(batchSizes, &batchSizesShape);
    diopiRequireTensor(ctx, &groupListTensor, &batchSizesShape, nullptr, diopi_dtype_int64, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnCumsumV2, ctx, batchSizes, 0, false, false, groupListTensor);

    std::vector<int64_t> groupList(batchSizesShape.data[0]);
    void* dataPtr = nullptr;
    diopiGetTensorData(groupListTensor, &dataPtr);
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));
    CALL_ACLRT(aclrtMemcpy(
        groupList.data(), sizeof(int64_t) * batchSizesShape.data[0], dataPtr, sizeof(int64_t) * batchSizesShape.data[0], ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<diopiConstTensorHandle_t> inputsVec = {inputs};
    std::vector<diopiConstTensorHandle_t> weightsVec = {weights};
    std::vector<diopiTensorHandle_t> outsVec = {outs};

    if (transInputs) {
        diopiTensorHandle_t inputsTrans = transposeTensor(ctx, inputs);
        std::vector<diopiTensorHandle_t> inputsTransVec = {inputsTrans};
        DIOPI_ASCEND_CALL_ACLNN(aclnnGroupedMatmulV2, ctx, inputsTransVec, weightsVec, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, 3, 2, outsVec);
    } else if (transWeights) {
        diopiTensorHandle_t weightsTrans = transposeTensor(ctx, weights);
        std::vector<diopiTensorHandle_t> weightsTransVec = {weightsTrans};
        DIOPI_ASCEND_CALL_ACLNN(aclnnGroupedMatmulV2, ctx, inputsVec, weightsTransVec, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, 3, 0, outsVec);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnGroupedMatmulV2, ctx, inputsVec, weightsVec, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, 3, 0, outsVec);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
