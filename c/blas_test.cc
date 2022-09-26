// from https://jott.live/code/blas_test.cc
// via https://news.ycombinator.com/item?id=29310509

// macos:
// clang++ blas_test.cc -framework Accelerate -std=c++11 -O3 -o blas_test

// linux:
// g++ blas_test.cc -lblas -std=c++11 -O3 -o blas_test

// run:
// ./blas_test 512 512 512 100 100

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << argv[0] << " M N K [iters] [reps]\n";
        return 1;
    }
    auto M = 1;
    auto N = 1;
    auto K = 1;
    {
        std::stringstream ss(argv[1]);
        ss >> M;
    }
    {
        std::stringstream ss(argv[2]);
        ss >> N;
    }
    {
        std::stringstream ss(argv[3]);
        ss >> K;
    }
    auto LDA = K;
    auto LDB = N;
    auto LDC = N;
    auto *A = (float *)malloc(sizeof(float) * M * K);
    auto *B = (float *)malloc(sizeof(float) * K * N);
    auto *C = (float *)malloc(sizeof(float) * M * N);
    for (auto i = 0; i < M * K; ++i) {
        A[i] = 0.7;
    }
    for (auto i = 0; i < N * K; ++i) {
        B[i] = 0.4;
    }
    auto iters = int(1e11 / (M * N * K));
    auto reps = 10;
    if (iters < 10) {
        iters = 10;
    }
    if (argc > 4) {
        std::stringstream ss(argv[4]);
        ss >> iters;
    }
    if (argc > 5) {
        std::stringstream ss(argv[5]);
        ss >> reps;
    }
    for (auto _ = 0; _ < reps; ++_) {
        for (auto i = 0; i < iters / 10; ++i) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A,
                        LDA, B, LDB, 0.0, C, LDC);
        }
        auto start = std::chrono::steady_clock::now();
        for (auto i = 0; i < iters; ++i) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A,
                        LDA, B, LDB, 0.0, C, LDC);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << 1.0 * M * N * K * 2 * iters / elapsed_seconds.count() / 1e9
                  << " gflops\n";
    }
    free(A);
    free(B);
    free(C);
}
