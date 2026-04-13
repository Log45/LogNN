#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("sm_%d%d", prop.major, prop.minor);
    return 0;
}
