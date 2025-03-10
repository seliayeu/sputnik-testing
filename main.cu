#include <cmath>
#include <iostream>
#include <fstream>
#include <sputnik/spmm/cuda_spmm.h>
#include <random>
#include <cuda_runtime_api.h> 
#include <cusparse.h>        

std::mt19937 gen;
std::random_device rd;

float* generateMatrix(int M, int N, float sparsity) {
    std::uniform_real_distribution<float> entryDist(-1, 1);
    std::uniform_real_distribution<float> choiceDist(0, 1);

    float* out = new float[M * N];
    for (int i = 0; i < M * N; ++i) {
        if (choiceDist(gen) < sparsity)
            continue;
        out[i] = entryDist(gen);
    }

    return out;
}

void convertToCSR(float*& values, int*& rowOffsets, int*& colIndices, int& nnz, int M, int N, const float* input) {
    rowOffsets = new int[M + 1];

    for (int i = 0; i < M * N; ++i) {
        if (input[i] != 0)
            nnz += 1;
    }
    
    values = new float[nnz];
    colIndices = new int[nnz];

    int ind = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (input[i * M + j] == 0)
                continue;
            values[ind] = input[i * M + j];
            colIndices[ind] = j;
        }
        rowOffsets[i + 1] = ind;
    }
}

void readDLMCMatrix(const std::string filename, float*& values, int*& rowOffsets, int*& colIndices, int& nnz, int& M, int& N) {
    std::ifstream matrixFile(filename);
    char comma;
    std::uniform_real_distribution<float> entryDist(-1, 1);

    matrixFile >> M >> comma >> N >> comma >> nnz;
    rowOffsets = new int[M + 1];
    colIndices = new int[nnz];
    values = new float[nnz];

    for (int i = 0; i < M + 1; ++i)
        matrixFile >> rowOffsets[i];
    for (int i = 0; i < nnz; ++i) {
        matrixFile >> colIndices[i];
        values[i] = entryDist(gen);
    }
}

int main() {
    gen = std::mt19937{rd()};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int M = 4096;
    int N = 4096;
    int K = 4096;
    int sparsity = 0.8;
    
    float* A = generateMatrix(M, K, sparsity);
    float* B = generateMatrix(K, N, 0);
    
    float* valuesA;
    int* rowOffsetsA;
    int* colIndicesA;
    int nnz = 0;

    int* rowIndices = new int[M];

    for (int i = 0; i < M; ++i)
        rowIndices[i] = i;

    convertToCSR(valuesA, rowOffsetsA, colIndicesA, nnz, M, K, A);

    float* d_valuesA;
    cudaMalloc(&d_valuesA, nnz * sizeof(float));
    int* d_colIndicesA;
    cudaMalloc(&d_colIndicesA, N * sizeof(int));
    int* d_rowOffsetsA;
    cudaMalloc(&d_rowOffsetsA, (M + 1) * sizeof(int));
    float* d_B;
    cudaMalloc(&d_B, M * N * sizeof(float));
    
    int* d_rowIndices;
    cudaMalloc(&d_rowIndices, M * sizeof(int));

    cudaMemcpy(d_valuesA, valuesA, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndicesA, colIndicesA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowOffsetsA, rowOffsetsA, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rowIndices, rowIndices, M * sizeof(int), cudaMemcpyHostToDevice);

    float* d_C;
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sputnik::CudaSpmm(M, K, N, nnz, d_rowIndices, d_valuesA, d_rowOffsetsA, d_colIndicesA, d_B, d_C, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << std::endl;

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

    cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaFree(d_valuesA);
    cudaFree(d_colIndicesA);
    cudaFree(d_rowOffsetsA);
    cudaFree(d_B);
    cudaFree(d_rowIndices);
    cudaFree(d_C);
    cudaFree(dBuffer);

    delete A;
    delete B;
    delete valuesA;
    delete rowOffsetsA;
    delete colIndicesA;

    return 0;
}
