#include<iostream>
#include<vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

__global__ void add_vec(int* a, int* b, int* c, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < n){
        c[tid] = a[tid] + b[tid];
    }

}

void vec(int n){
    vector<int> one(n);
    vector<int> two(n);
    vector<int> sum_of_two_cpu(n);
    vector<int> sum_of_two(n);

    int *dev_a, *dev_b, *dev_c;

    for(int i=0; i<n; i++){
        one[i] = i+1;
        two[i] = i+2;
    }

    auto start_cpu = high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        sum_of_two_cpu[i] = one[i] + two[i];
    }

    auto end_cpu = high_resolution_clock::now();

    auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);


    cudaMalloc((void**) &dev_a, n*sizeof(int));
    cudaMalloc((void**) &dev_b, n*sizeof(int));
    cudaMalloc((void**) &dev_c, n*sizeof(int));

    cudaMemcpy(dev_a, one.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, two.data(), n*sizeof(int) , cudaMemcpyHostToDevice);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_vec<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b, dev_c, n);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float duration_gpu = 0;
    cudaEventElapsedTime(&duration_gpu, start_gpu, stop_gpu);


    cudaMemcpy(sum_of_two.data(), dev_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cout << "Time taken by CPU: " << duration_cpu.count() << " microseconds" << endl;
    cout << "Time taken by GPU: " << duration_gpu << " milliseconds" << endl;

    
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

}


int main(){

    vector<int> a ={100, 1000, 100000, 1000000};

    for(int i=0; i<a.size(); i++){
        cout << "Vector size: " << a[i] << endl;
        vec(a[i]);
        cout << endl;
    }
    return 0;
}