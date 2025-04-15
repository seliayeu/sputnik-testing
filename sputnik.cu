#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sputnik/spmm/cuda_spmm.h>
#include <random>
#include <cuda_runtime_api.h> 
#include <cusparse.h>        
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int* rowIndices = new int[M];

    for (int i = 0; i < M; ++i)
        rowIndices[i] = i;

    
    N = M;
    float* B = generateMatrix(K, N, 0);

    float* d_valuesA;
    gpuErrchk(cudaMalloc(&d_valuesA, nnz * sizeof(float)));
    int* d_colIndicesA;
    gpuErrchk(cudaMalloc(&d_colIndicesA, nnz * sizeof(int)));
    int* d_rowOffsetsA;
    gpuErrchk(cudaMalloc(&d_rowOffsetsA, (M + 1) * sizeof(int)));

    float* d_B;
    gpuErrchk(cudaMalloc(&d_B, K * N * sizeof(float)));
    
    int* d_rowIndices;
    gpuErrchk(cudaMalloc(&d_rowIndices, M * sizeof(int)));

    gpuErrchk(cudaMemcpy(d_valuesA, valuesA, nnz * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_colIndicesA, colIndicesA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rowOffsetsA, rowOffsetsA, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_rowIndices, rowIndices, M * sizeof(int), cudaMemcpyHostToDevice));

    float* d_C;
    gpuErrchk(cudaMalloc(&d_C, M * N * sizeof(float)));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));
    gpuErrchk(sputnik::CudaSpmm(M, K, N, nnz, d_rowIndices, d_valuesA, d_rowOffsetsA, d_colIndicesA, d_B, d_C, stream));
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << milliseconds << std::endl;

    cudaFree(d_valuesA);
    cudaFree(d_colIndicesA);
    cudaFree(d_rowOffsetsA);
    cudaFree(d_B);
    cudaFree(d_rowIndices);
    cudaFree(d_C);

    delete B;
    delete valuesA;
    delete rowOffsetsA;
    delete colIndicesA;

    return 0;
}
