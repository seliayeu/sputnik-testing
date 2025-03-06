#include <cmath>
#include <iostream>
#include <sputnik/spmm/cuda_spmm.h>
#include <random>

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

int main() {
    gen = std::mt19937{rd()};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int N = 1024;
    int K = 1024;
    int M = 1024;
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

    // std::cout <<"wowzer" << std::endl;
    
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

    sputnik::CudaSpmm(M, K, N, nnz, d_rowIndices, d_valuesA, d_rowOffsetsA, d_colIndicesA, d_B, d_C, stream);

    // std::cout <<"yay" << std::endl;
    return 0;
}
