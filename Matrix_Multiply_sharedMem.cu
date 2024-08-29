#include<iostream>

#define TILE_SIZE 16

using namespace std;

__global__ void MatrixMultiShared(float* A, float* B, float* C, int N){
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x + blockIdx.x * TILE_SIZE;

    float val = 0.0f;

    for(int i =0; i < (N + TILE_SIZE -1)/ TILE_SIZE; i++){
        if(row < N && (i * TILE_SIZE + threadIdx.x) < N){
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
        }
        else{
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }


        if(col < N && (i * TILE_SIZE + threadIdx.y) < N){
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        }
        else{
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int j=0; j<TILE_SIZE; j++){
            val+= tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < N && col < N){
        C[row * N + col] = val;
    }
}


int main(){
    int N = 1024;
    int size = N * N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*) malloc(size);

    for(int i=0; i< N*N; i++){
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1)/ TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE );

    MatrixMultiShared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++){
    //     for(int j=0; j< N; j++){
    //         cout<<h_c[i * N + j]<< " ";
    //     }
    //     cout<<endl;
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;

}