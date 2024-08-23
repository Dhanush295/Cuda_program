#include<iostream>
#define n 10

using namespace std;
const int threadperblock = 256;


__global__ void dot_product(int *a, int *b, int *c){
    __shared__ float cache[threadperblock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp =0;

    while(tid < n){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x/2;
    while(i!= 0){
        if(cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0){
        c[blockIdx.x] = cache[0];
        }
}

int main(){

    int a[n], b[n], c[n], cpu_stored , stored_dot_val;
    int *dev_a, *dev_b, *dev_c;

    for(int i=0; i<n; i++){
        a[i] = i+2;
        b[i] = i+1;
    }

    cpu_stored = 0;

    for(int i=0; i<n; i++){
        cpu_stored += a[i] * b[i];
    }

    cout<<"CPU STored Value: "<<cpu_stored<<endl;



    cudaMalloc( (void **) &dev_a, n*sizeof(int) );
    cudaMalloc( (void **) &dev_b, n*sizeof(int) );
    cudaMalloc((void **) &dev_c, n*sizeof(int));

    cudaMemcpy(dev_a, a, n*sizeof(n), cudaMemcpyHostToDevice );
    cudaMemcpy(dev_b, b, n*sizeof(n), cudaMemcpyHostToDevice );


    int blockpergrid = (n + threadperblock -1)/ threadperblock;

    dot_product<<<blockpergrid, threadperblock>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&c, dev_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    stored_dot_val = 0;

    for( int i=0; i<blockpergrid; i++){
        stored_dot_val += c[i];
    }

    cout<<"Stored Value is: "<< stored_dot_val<<endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}