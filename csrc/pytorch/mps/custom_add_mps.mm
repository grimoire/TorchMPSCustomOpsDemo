#include <torch/extension.h>
#include "MPSStream.h"
using at::Tensor;

// utils
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor){
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

Tensor custom_add(const Tensor &a, const Tensor &b){
    auto stream = at::mps::getCurrentMPSStream();
    return a+b;
}