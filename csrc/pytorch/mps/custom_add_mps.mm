#include <torch/extension.h>
#include "MPSStream.h"
#include "MPSLibrary.h"
#include "MPSUtils.h"
#include <cassert>
using at::Tensor;

void CustomAddImpl(Tensor a, Tensor b, Tensor c, size_t num_elements){
    // get stream
    auto stream = at::mps::getCurrentMPSStream();
    auto library_manager = MPSLibraryManager::getInstance();
    auto library = library_manager->getLibrary("_metal_kernel.metallib");
    auto func_pso = library->getComputePipelineState("custom_add_impl");
    
    // create command buffer and encoder
    MTLCommandBuffer_t command_buffer = stream->commandBuffer();
    MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

    //set pso and buffer
    setMTLArgs(compute_encoder, func_pso, a, b, c);

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

    size_t num_elements = a.numel();
    CustomAddImpl(a, b, ret, num_elements);
    return ret;
}