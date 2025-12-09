/**
 * cache_probe.c - L1/L2/L3 cache contention probe
 *
 * Stresses the cache hierarchy via repeated accesses to a large working set.
 * Uses pointer-chasing to avoid prefetching and force cache misses.
 *
 * Outputs: CSV with timestamp, probe name, iteration, and cycles per iteration.
 */
#include "common.h"

// Cache probe configuration
#define CACHE_LINE_SIZE 64
#define WORKING_SET_SIZE (8 * 1024 * 1024)  // 8 MB working set (larger than L3 on many CPUs)
#define ACCESS_COUNT 1024  // Number of accesses per iteration

typedef struct node {
    struct node* next;
    char padding[CACHE_LINE_SIZE - sizeof(struct node*)];
} node_t;

// Initialize pointer-chasing linked list with pseudo-random order
static void init_pointer_chain(node_t* nodes, size_t count) {
    // Simple pseudo-random permutation using modular arithmetic
    size_t stride = 179;  // Prime number for pseudo-random access pattern
    size_t idx = 0;
    
    for (size_t i = 0; i < count - 1; i++) {
        size_t next_idx = (idx + stride) % count;
        nodes[idx].next = &nodes[next_idx];
        idx = next_idx;
    }
    // Close the loop
    nodes[idx].next = &nodes[0];
}

int main(int argc, char** argv) {
    probe_args_t args = parse_probe_args(argc, argv, "cache");
    
    // Allocate cache-line-aligned working set
    size_t num_nodes = WORKING_SET_SIZE / sizeof(node_t);
    node_t* nodes = (node_t*)aligned_alloc_safe(CACHE_LINE_SIZE, num_nodes * sizeof(node_t));
    
    // Initialize pointer-chasing chain
    init_pointer_chain(nodes, num_nodes);
    
    // Warmup
    volatile node_t* ptr = &nodes[0];
    for (int i = 0; i < WARMUP_ITERS; i++) {
        for (int j = 0; j < ACCESS_COUNT; j++) {
            ptr = ptr->next;
        }
    }
    
    // Print CSV header
    printf(CSV_HEADER);
    fflush(stdout);
    
    // Main measurement loop
    for (int iter = 0; iter < args.iters; iter++) {
        uint64_t cycles;
        ptr = &nodes[0];
        
        MEASURE_CYCLES(cycles, {
            for (int j = 0; j < ACCESS_COUNT; j++) {
                ptr = ptr->next;
            }
            COMPILER_BARRIER();
        });
        
        print_csv_row("cache", iter, cycles);
        
        // Periodic flush to avoid buffer buildup
        if (iter % 100 == 0) {
            fflush(stdout);
        }
    }
    
    // Use ptr to prevent optimization
    if (ptr == NULL) {
        printf("# unreachable\n");
    }
    
    free(nodes);
    return 0;
}
