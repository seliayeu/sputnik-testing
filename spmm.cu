#include <cmath>
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
    
    N = M;
    float* B = generateMatrix(K, N, 0);

    float* d_valuesA;
    int* d_colIndicesA;
    int* d_rowOffsetsA;
    gpuErrchk(cudaMalloc(&d_valuesA, nnz * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_colIndicesA, nnz * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_rowOffsetsA, (M + 1) * sizeof(int)));

    float* d_B;
    gpuErrchk(cudaMalloc(&d_B, K * N * sizeof(float)));
    
    gpuErrchk(cudaMemcpy(d_valuesA, valuesA, nnz * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_colIndicesA, colIndicesA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rowOffsetsA, rowOffsetsA, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float* d_C;
    gpuErrchk(cudaMalloc(&d_C, M * N * sizeof(float)));

    cudaEvent_t start, stop;

    // code from cuSPARSE example
    
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float alpha           = 1.0f;
    float beta            = 0.0f;

    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, M, K, nnz, d_rowOffsetsA, d_colIndicesA, d_valuesA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnMat(&matB, K, N, K, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, M, N, M, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseSpMM_bufferSize(     handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSpMM_preprocess(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));
    cusparseStatus_t status;

    status = cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf ("CUSPARSE kernel failed\n");
        return EXIT_FAILURE;
    }
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << milliseconds << std::endl;

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaFree(d_valuesA);
    cudaFree(d_colIndicesA);
    cudaFree(d_rowOffsetsA);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(dBuffer);

    delete B;
    delete valuesA;
    delete rowOffsetsA;
    delete colIndicesA;

    return 0;
}
