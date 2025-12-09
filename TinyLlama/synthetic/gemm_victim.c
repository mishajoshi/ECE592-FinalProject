#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

// Simple block size for tiling (cache friendly)
#define BLOCK_SIZE 64

// Matrix-Vector Multiplication (GEMV) to simulate Token Generation phase
// This is O(N^2) memory access, matching the bandwidth-bound nature of LLM inference
void gemv_blocked(float *A, float *B, float *C, int N) {
    // A is Weight Matrix [N x N]
    // B is Input Vector [N]
    // C is Output Vector [N]
    
    // Simple blocked GEMV
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            
            int i_max = (i + BLOCK_SIZE > N) ? N : i + BLOCK_SIZE;
            int j_max = (j + BLOCK_SIZE > N) ? N : j + BLOCK_SIZE;

            for (int ii = i; ii < i_max; ii++) {
                float sum = C[ii];
                for (int jj = j; jj < j_max; jj++) {
                    sum += A[ii * N + jj] * B[jj];
                }
                C[ii] = sum;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dim> [iters]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int iters = (argc > 2) ? atoi(argv[2]) : 10;

    printf("Initializing Synthetic GEMV Victim: Dim=%d, Iters=%d\n", N, iters);
    printf("Memory footprint: %.2f MB\n", (1.0 * N * N * sizeof(float)) / (1024*1024));

    // Allocate aligned memory
    float *A, *B, *C;
    posix_memalign((void**)&A, 64, N * N * sizeof(float)); // Matrix
    posix_memalign((void**)&B, 64, N * sizeof(float));     // Vector
    posix_memalign((void**)&C, 64, N * sizeof(float));     // Vector

    // Initialize
    for (int i = 0; i < N * N; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N; i++) B[i] = (float)rand() / RAND_MAX;

    // Run workload
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // TinyLLaMA has 22 layers. 
    // Each layer has roughly 7 major GEMVs (Q, K, V, Output, Gate, Up, Down).
    int num_layers = 22;
    int gemms_per_layer = 7;

    // Busy wait to simulate compute overhead (Attention, LayerNorm, Activation)
    // This keeps the CPU busy but doesn't thrash L3, spacing out the GEMV bursts.
    // Calibrated to match the 0.02 Hz frequency of the real victim.
    long busy_cycles = 5000000; 

    for (int i = 0; i < iters; i++) {
        // Simulate one token generation (forward pass through layers)
        for (int l = 0; l < num_layers; l++) {
            for (int g = 0; g < gemms_per_layer; g++) {
                 gemv_blocked(A, B, C, N);
            }
            
            // Simulate compute overhead (CPU bound, L1 bound)
            volatile int k = 0;
            for(long b=0; b<busy_cycles; b++) {
                k += b;
            }
        }
        
        // Prevent optimization
        if (C[0] > 1000000) printf("."); 
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    
    printf("\nDone. Time: %.2fs\n", time_taken);

    free(A);
    free(B);
    free(C);
    return 0;
}
