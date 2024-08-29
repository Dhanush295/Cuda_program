#include<iostream>

#define N 1024

using namespace std;

__global__ void MatrixMultiDevice(float* d_a, float* d_b, float* d_c, int width){

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < width && col < width){
        float val = 0.0f;

        for(int k=0; k< width; k++){
            val += d_a[row * width + k] * d_b[k * width + col];
        }

        d_c[row* width + col] = val;
    }
}


void matrixMultiHost(float* h_a, float* h_b, float* h_c, int width){

    int size = width * width * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , h_b, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1)/ dimBlock.x, (width + dimBlock.y - 1)/ dimBlock.y);

    MatrixMultiDevice<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


}

int main(){
    int width = N;
    int size = width * width * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    for(int i=0; i< width* width; i++){
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    matrixMultiHost(h_a, h_b, h_c, width);

    // for(int i=0; i< width; i++){
    //     for(int j=0; j< width; j++){
    //         cout<<h_c[i * width + j] <<" ";
    //     }
    //     cout<<endl;
    // }

    free(h_a);
    free(h_b);
    free(h_c);
}