#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <xmmintrin.h>

constexpr int SIZE = 1024; // Размер матрицы, кратный 4 для SSE
using Matrix = std::vector<std::vector<float>>;

// Наивное умножение матриц
void multiply_naive(const Matrix &A, const Matrix &B, Matrix &C) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// OpenMP с двойным коллапсом циклов
void multiply_omp(const Matrix &A, const Matrix &B, Matrix &C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// SSE с плоскими массивами и корректной обработкой
void multiply_sse_intrinsics(const float *A, const float *B, float *C) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < SIZE; k += 4) {
                __m128 a_line = _mm_loadu_ps(&A[i * SIZE + k]);
                __m128 b_line = _mm_set_ps(B[(k + 3) * SIZE + j],
                                           B[(k + 2) * SIZE + j],
                                           B[(k + 1) * SIZE + j],
                                           B[k * SIZE + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a_line, b_line));
            }
            float result[4];
            _mm_storeu_ps(result, sum);
            C[i * SIZE + j] = result[0] + result[1] + result[2] + result[3];
        }
    }
}

// OpenMP + SSE с устранением конфликта данных
void multiply_omp_sse(const float *A, const float *B, float *C) {
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < SIZE; k += 4) {
                __m128 a_line = _mm_loadu_ps(&A[i * SIZE + k]);
                __m128 b_line = _mm_set_ps(B[(k + 3) * SIZE + j],
                                           B[(k + 2) * SIZE + j],
                                           B[(k + 1) * SIZE + j],
                                           B[k * SIZE + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a_line, b_line));
            }
            float result[4];
            _mm_storeu_ps(result, sum);
            C[i * SIZE + j] = result[0] + result[1] + result[2] + result[3];
        }
    }
}

int main() {
    Matrix A(SIZE, std::vector<float>(SIZE, 1.0f));
    Matrix B(SIZE, std::vector<float>(SIZE, 1.0f));
    Matrix C(SIZE, std::vector<float>(SIZE, 0.0f));

    std::vector<float> A_flat(SIZE * SIZE, 1.0f);
    std::vector<float> B_flat(SIZE * SIZE, 1.0f);
    std::vector<float> C_flat(SIZE * SIZE, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    multiply_naive(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Naive multiply time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    multiply_omp(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OMP multiply time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    multiply_sse_intrinsics(A_flat.data(), B_flat.data(), C_flat.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "SSE multiply time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    multiply_omp_sse(A_flat.data(), B_flat.data(), C_flat.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OMP + SSE multiply time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";

    return 0;
}
