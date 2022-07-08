#include <torch/extension.h>
#include "MPSStream.h"
#include "MPSLibrary.h"
#include <cassert>
using at::Tensor;


#ifdef __OBJC__
typedef id<MTLBuffer> MTLBuffer_t;
#else
typedef void* MTLBuffer;
typedef void* MTLBuffer_t;
#endif

// utils
static inline MTLBuffer_t getMTLBufferStorage(const at::Tensor& tensor){
    return __builtin_bit_cast(MTLBuffer_t, tensor.storage().data());
}

void CustomAddImpl(MTLBuffer_t a, MTLBuffer_t b, MTLBuffer_t c, size_t num_elements){
    // get stream
    auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    auto library = library_manager->getLibrary("_metal_kernel.metallib");
    auto func_pso = library->getComputePipelineState("custom_add_impl");
    
    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];

    //set pso and buffer
    [compute_encoder setComputePipelineState:func_pso];
    [compute_encoder setBuffer:a offset:0 atIndex:0];
    [compute_encoder setBuffer:b offset:0 atIndex:1];
    [compute_encoder setBuffer:c offset:0 atIndex:2];

    // set grid size
    MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
    NSUInteger thread_group_size_x = func_pso.maxTotalThreadsPerThreadgroup;
    if (thread_group_size_x > num_elements){
        thread_group_size_x = num_elements;
    }
    MTLSize thread_group_size = MTLSizeMake(thread_group_size_x, 1, 1);

    // encoding
    [compute_encoder dispatchThreads: grid_size threadsPerThreadgroup: thread_group_size];
    [compute_encoder endEncoding];

    // commit, not sure if flush is required
    stream->commit(true);
}

Tensor custom_add(const Tensor &a, const Tensor &b){
    Tensor ret = at::empty_like(a);

    auto buffer_a = getMTLBufferStorage(a);
    auto buffer_b = getMTLBufferStorage(b);
    auto buffer_ret = getMTLBufferStorage(ret);

    size_t num_elements = a.numel();
    CustomAddImpl(buffer_a, buffer_b, buffer_ret, num_elements);
    return ret;
}