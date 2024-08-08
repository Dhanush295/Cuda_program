#include<iostream>
#include<cuda_runtime.h>

#define N  10
using namespace std;


__global__ void add( int* a, int* b, int* c){

    int tid = blockIdx.x;

    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }

}

int main(){
    int a[N] , b[N], c[N];
    
    int *dev_a , *dev_b , *dev_c;

    for(int i=0; i<N; i++ ){
        a[i] = i+1;
        b[i] = i*i;
    }

    cudaMalloc((void **) &dev_a , N*sizeof(int));
    cudaMalloc((void **) &dev_b , N*sizeof(int));
    cudaMalloc((void **) &dev_c , N*sizeof(int));

    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N,1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<N ;i++){
        cout<<"A + B "<<c[i]<<"\n";
        
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;

}