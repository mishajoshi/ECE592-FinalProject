/**
 * common.h - Shared utilities for SMT contention probes
 *
 * Provides timing primitives (RDTSC), CSV output helpers, and memory allocation macros.
 */
#ifndef COMMON_H
#define COMMON_H

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// RDTSC inline assembly for cycle counting
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Serialize before RDTSC (CPUID or LFENCE)
static inline void serialize(void) {
    __asm__ __volatile__ ("lfence" ::: "memory");
}

// Measure cycles for a code block
#define MEASURE_CYCLES(cycles_var, code_block) do { \
    uint64_t __start, __end; \
    serialize(); \
    __start = rdtsc(); \
    code_block \
    serialize(); \
    __end = rdtsc(); \
    cycles_var = __end - __start; \
} while(0)

// Get current timestamp in nanoseconds
static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Memory fence
static inline void mfence(void) {
    __asm__ __volatile__ ("mfence" ::: "memory");
}

// Prevent compiler optimizations
#define COMPILER_BARRIER() __asm__ __volatile__ ("" ::: "memory")

// CSV header format for probe output
#define CSV_HEADER "ts_ns,probe,iter,cycles\n"

// Print CSV row
static inline void print_csv_row(const char* probe_name, uint64_t iter, uint64_t cycles) {
    uint64_t ts = get_timestamp_ns();
    printf("%lu,%s,%lu,%lu\n", ts, probe_name, iter, cycles);
}

// Allocate aligned memory (for cache line / page alignment)
static inline void* aligned_alloc_safe(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Parse command line args for standard probe interface
typedef struct {
    int cpu;
    int iters;
} probe_args_t;

static inline probe_args_t parse_probe_args(int argc, char** argv, const char* probe_name __attribute__((unused))) {
    probe_args_t args = {-1, 2000};  // defaults
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cpu_id> [iterations]\n", argv[0]);
        fprintf(stderr, "Example: %s 3 5000\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    args.cpu = atoi(argv[1]);
    if (argc >= 3) {
        args.iters = atoi(argv[2]);
    }
    
    if (args.iters <= 0) {
        fprintf(stderr, "Invalid iteration count: %d\n", args.iters);
        exit(EXIT_FAILURE);
    }
    
    return args;
}

// Warmup iterations (to be discarded)
#define WARMUP_ITERS 100

#endif // COMMON_H
