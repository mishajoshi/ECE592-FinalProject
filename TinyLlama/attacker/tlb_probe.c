/**
 * tlb_probe.c - TLB (Translation Lookaside Buffer) contention probe
 *
 * Stresses the TLB by accessing memory with large strides across many pages.
 * Each access targets a different page to maximize TLB pressure.
 *
 * Outputs: CSV with timestamp, probe name, iteration, and cycles per iteration.
 */
#include "common.h"
#include <sys/mman.h>

// TLB probe configuration
#define PAGE_SIZE 4096
#define NUM_PAGES 2048  // Access 2048 pages (8 MB total, exceeds typical TLB capacity)
#define ACCESSES_PER_ITER 512  // Number of page accesses per iteration

int main(int argc, char** argv) {
    probe_args_t args = parse_probe_args(argc, argv, "tlb");
    
    // Allocate large memory region
    size_t total_size = NUM_PAGES * PAGE_SIZE;
    char* memory = (char*)mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (memory == MAP_FAILED) {
        perror("mmap");
        return EXIT_FAILURE;
    }
    
    // Touch each page to ensure it's mapped
    for (size_t i = 0; i < NUM_PAGES; i++) {
        memory[i * PAGE_SIZE] = (char)i;
    }
    
    // Warmup
    volatile int dummy = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        for (int j = 0; j < ACCESSES_PER_ITER; j++) {
            size_t page_idx = (j * 17) % NUM_PAGES;  // Pseudo-random page access
            dummy += memory[page_idx * PAGE_SIZE];
        }
    }
    
    // Print CSV header
    printf(CSV_HEADER);
    fflush(stdout);
    
    // Main measurement loop
    for (int iter = 0; iter < args.iters; iter++) {
        uint64_t cycles;
        dummy = 0;
        
        MEASURE_CYCLES(cycles, {
            for (int j = 0; j < ACCESSES_PER_ITER; j++) {
                // Access different pages in pseudo-random order
                size_t page_idx = (j * 17 + iter * 3) % NUM_PAGES;
                dummy += memory[page_idx * PAGE_SIZE];
            }
            COMPILER_BARRIER();
        });
        
        print_csv_row("tlb", iter, cycles);
        
        // Periodic flush
        if (iter % 100 == 0) {
            fflush(stdout);
        }
    }
    
    // Use dummy to prevent optimization
    if (dummy > 1000000) {
        printf("# unreachable\n");
    }
    
    munmap(memory, total_size);
    return 0;
}
