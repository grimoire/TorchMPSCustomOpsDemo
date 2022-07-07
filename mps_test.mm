#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>

typedef id<MTLDevice> MTLDevice_t;
int main(){
    MTLDevice_t device = MTLCreateSystemDefaultDevice();
    printf("mps_test finish\n");
    return 0;
}