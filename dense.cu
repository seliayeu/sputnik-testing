#include <cmath>
#include <iostream>
#include <fstream>
#include <sputnik/spmm/cuda_spmm.h>
#include <random>
#include <cuda_runtime_api.h> 
#include "cublas_v2.h"
#include "utils.h"

std::mt19937 gen;
std::random_device rd;

int main(int argc, char* argv[]) {
    gen = std::mt19937{rd()};

    std::string matrixPath;
    int sparsity;

    int M, N, K;

    float* valuesA;
    int* rowOffsetsA;
    int* colIndicesA;
    int nnz;

    if (argc == 2) {
        matrixPath = argv[1];
        readDLMCMatrix(matrixPath, valuesA, rowOffsetsA, colIndicesA, nnz, M, K);
    } else if (argc == 5) {
        M = std::stoi(argv[1]);
        K = std::stoi(argv[2]);
        N = std::stoi(argv[3]);;
        sparsity = std::stof(argv[4]);
        float* A = generateMatrix(M, K, sparsity);
        convertToCSR(valuesA, rowOffsetsA, colIndicesA, nnz, M, K, A);
        delete A;
    } else {
        std::cerr << "Invalid number of arguments." << std::endl;
    }

    N = M;
    B = generateMatrix(K, N, 0);

    float* d_A;
    float* d_B;
    float* d_C;

    float alpha = 1;
    float beta = 0;

    gpuErrchk(cudaMalloc(&d_A, M * K * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, K * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, M * N * sizeof(float)));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
            d_A, CUDA_R_32F, M, 
            d_B, CUDA_R_32F, K, 
            &beta,
            d_C, CUDA_R_32F, M, 
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << milliseconds << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    delete A;
    delete B;
    delete valuesA;
    delete rowOffsetsA;
    delete colIndicesA;

    return 0;
}
