#include <iostream>
#include <cstdlib>
using namespace std; 
#define N 3  // Matrix size (NxN)

__global__ void matrixMulKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

void printMatrix(int* mat, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << mat[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int size = N * N * sizeof(int);
    int h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    int block_x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int block_y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y ; 
    cout<<block_x<<endl;
    cout<<block_y<<endl;
    dim3 numBlocks(block_x, block_y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output
    std::cout << "Matrix A:\n";
    printMatrix(h_A, N);

    std::cout << "\nMatrix B:\n";
    printMatrix(h_B, N);

    std::cout << "\nMatrix C (A x B):\n";
    printMatrix(h_C, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
