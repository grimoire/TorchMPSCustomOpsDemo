#include "MPSLibrary.h"
#include "MPSDevice.h"
#include <c10/util/CallOnce.h>

static std::unique_ptr<MPSLibraryManager> mps_library_manager;
static c10::once_flag mpsdev_init;

MPSLibraryManager* MPSLibraryManager::getInstance() {
    c10::call_once(mpsdev_init, []{
        mps_library_manager = std::unique_ptr<MPSLibraryManager>(new MPSLibraryManager());
    });
    return mps_library_manager.get();
}

MPSLibraryManager::~MPSLibraryManager(){}

MPSLibraryManager::MPSLibraryManager() {}


MPSLibrary* MPSLibraryManager::getLibrary(const std::string& library_url){
    if(_library_map.find(library_url)!=_library_map.end()){
        return _library_map[library_url].get();
    }
    _library_map.emplace(std::make_pair(library_url, std::make_unique<MPSLibrary>(library_url)));
    return _library_map[library_url].get();
}

MPSLibrary::MPSLibrary(const std::string& library_url){

    @autoreleasepool{
    NSError* error = nil;

    // load library and func
    NSString* utl_str = [NSString stringWithCString: library_url.c_str()];
    NSURL* metal_url = [NSURL fileURLWithPath: utl_str];
    _library = [at::mps::MPSDevice::getInstance()->device() newLibraryWithURL: metal_url error:&error];
    if(_library == nil){
        NSLog(@"Failed to find library, error %@.", error);
        exit(1);
    }
    }
}

MPSLibrary::~MPSLibrary(){
    [_library release];
    _library = nil;
}

MTLComputePipelineState_t MPSLibrary::getComputePipelineState(const std::string& function_name){
    if(_pso_map.find(function_name)!=_pso_map.end()){
        return _pso_map[function_name];
    }

    MTLComputePipelineState_t pso;
    @autoreleasepool{
        NSError* error = nil;

        // create function
        NSString* function_name_str = [NSString stringWithCString: function_name.c_str()];
        id<MTLFunction> func = [_library newFunctionWithName: function_name_str];
        if(func == nil){
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            exit(1);
        }
        // create pipeline
        pso = [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction: func error:&error];
        _pso_map.emplace(std::make_pair(function_name, pso));
    }
    return _pso_map[function_name];
}