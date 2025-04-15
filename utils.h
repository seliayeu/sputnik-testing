#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <sputnik/spmm/cuda_spmm.h>
#include <random>
#include <cuda_runtime_api.h> 
#include <cusparse.h>        

extern std::mt19937 gen;
extern std::random_device rd;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float* generateMatrix(int M, int N, float sparsity) {
    std::uniform_real_distribution<float> entryDist(-1, 1);
    std::uniform_real_distribution<float> choiceDist(0, 1);

    std::cout << sparsity << std::endl;
    float* out = new float[M * N];
    for (int i = 0; i < M * N; ++i) {
        float rand = choiceDist(gen);
        /*std::cout << (rand < sparsity) << std::endl;*/
        if (rand < sparsity)
            out[i] = 0;
        else
            out[i] = entryDist(gen);
    }

    return out;
}

void convertToCSR(float*& values, int*& rowOffsets, int*& colIndices, int& nnz, int M, int N, const float* input) {
    rowOffsets = new int[M + 1];
    nnz = 0;

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

void convertCSRToDense(float*& out, int M, int N, float*& values, int*& rowOffsets, int*& colIndices, int& nnz) {
    out = new float[M * N];
    for (int i = 1; i < M + 1; ++i)
        for (int j = rowOffsets[i - 1]; j < rowOffsets[i]; ++j)
            out[(i - 1) * N + colIndices[j]] = values[j];
}


