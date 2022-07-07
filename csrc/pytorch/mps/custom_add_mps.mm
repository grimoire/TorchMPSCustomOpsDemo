#include <torch/extension.h>
#include "MPSStream.h"
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

class MPSCustomAddImpl{
public:
    MPSCustomAddImpl(const MPSCustomAddImpl&) = delete;
    MPSCustomAddImpl& operator=(const MPSCustomAddImpl&)=delete;
    MPSCustomAddImpl(MPSCustomAddImpl &&)=delete;
    MPSCustomAddImpl& operator=(MPSCustomAddImpl &&)=delete;

    static auto& get_instance(){
        static MPSCustomAddImpl instance;
        return instance;
    }

    void send_compute_command(MTLBuffer_t a, MTLBuffer_t b, MTLBuffer_t c, size_t num_elements){
        // create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];

        //set pso and buffer
        [compute_encoder setComputePipelineState:_func_pso];
        [compute_encoder setBuffer:a offset:0 atIndex:0];
        [compute_encoder setBuffer:b offset:0 atIndex:1];
        [compute_encoder setBuffer:c offset:0 atIndex:2];

        // set grid size
        MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
        NSUInteger thread_group_size_x = _func_pso.maxTotalThreadsPerThreadgroup;
        if (thread_group_size_x > num_elements){
            thread_group_size_x = num_elements;
        }
        MTLSize thread_group_size = MTLSizeMake(thread_group_size_x, 1, 1);

        // encoding
        [compute_encoder dispatchThreads: grid_size threadsPerThreadgroup: thread_group_size];
        [compute_encoder endEncoding];

        // commit
        [command_buffer commit];

        // wait until finish, not necessary?
        // [command_buffer waitUntilCompleted];
    }
private:
    MPSCustomAddImpl(){
        // get stream
        auto stream = at::mps::getCurrentMPSStream();
        auto device = stream->device();
        NSError* error = nil;

        // load library and func
        id<MTLLibrary> metal_lib = [device newLibraryWithFile:@"_metal_kernel.metallib" error:&error];
        if(metal_lib == nil){
            NSLog(@"Failed to find library, error %@.", error);
            exit(1);
        }

        id<MTLFunction> func = [metal_lib newFunctionWithName:@"custom_add_impl"];
        if(func == nil){
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            exit(1);
        }

        // create pipeline
        _func_pso = [device newComputePipelineStateWithFunction: func error:&error];
        
        // get command queue
        _command_queue = stream->commandQueue();
    }

private:
    id<MTLComputePipelineState> _func_pso;
    id<MTLCommandQueue> _command_queue;

};

Tensor custom_add(const Tensor &a, const Tensor &b){
    auto& impl = MPSCustomAddImpl::get_instance();
    Tensor ret = at::empty_like(a);

    auto buffer_a = getMTLBufferStorage(a);
    auto buffer_b = getMTLBufferStorage(b);
    auto buffer_ret = getMTLBufferStorage(ret);

    size_t num_elements = a.numel();
    impl.send_compute_command(buffer_a, buffer_b, buffer_ret, num_elements);
    return ret;
}