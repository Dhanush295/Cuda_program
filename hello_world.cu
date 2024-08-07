#include<iostream>
#include<cuda_runtime.h>

__global__ void hello_cuda(){
    printf("Hello from GPU \n");
}



int main(){

    std::cout<<"Hello from CPU! \n";
    hello_cuda<<<1,1>>>();
    
    return 0;

}
