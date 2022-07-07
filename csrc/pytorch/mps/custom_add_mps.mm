#include <torch/extension.h>
#include "MPSStream.h"
#include <cassert>
using at::Tensor;

// utils
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor){
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
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
private:
    MPSCustomAddImpl(){
        // get stream
        auto stream = at::mps::getCurrentMPSStream();
        auto device = stream->device();
        NSError* error = nil;

        // load library and func
        id<MTLLibrary> default_lib = [device newDefaultLibrary];
        assert(default_lib != nil);

        id<MTLFunction> func = [default_lib newFunctionWithName:@"custom_add_impl"];
        assert(func != nil);

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

    return a+b;
}