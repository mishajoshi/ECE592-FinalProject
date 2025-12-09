/**
 * pht_probe.c - Pattern History Table (PHT) contention probe
 *
 * Stresses the PHT by executing many conditional branches with varying patterns.
 * Uses different sequences (predictable vs unpredictable) to maximize branch predictor pressure.
 *
 * Outputs: CSV with timestamp, probe name, iteration, and cycles per iteration.
 */
#include "common.h"

// PHT probe configuration
#define NUM_BRANCHES 512  // Number of conditional branches per iteration
#define PATTERN_SIZE 32   // Size of repeating pattern

// Different branch patterns
static int patterns[4][PATTERN_SIZE] = {
    // Pattern 0: Alternating (010101...)
    {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1},
    // Pattern 1: Mostly taken (11110111...)
    {1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1},
    // Pattern 2: Complex repeating pattern
    {1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,1},
    // Pattern 3: Pseudo-random (hard to predict)
    {1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,1,0,1,1,0}
};

static volatile int branch_counter = 0;

int main(int argc, char** argv) {
    probe_args_t args = parse_probe_args(argc, argv, "pht");
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        for (int j = 0; j < NUM_BRANCHES; j++) {
            int pattern_idx = (j / PATTERN_SIZE) % 4;
            int pos = j % PATTERN_SIZE;
            if (patterns[pattern_idx][pos]) {
                branch_counter++;
            } else {
                branch_counter--;
            }
        }
    }
    
    // Print CSV header
    printf(CSV_HEADER);
    fflush(stdout);
    
    // Main measurement loop
    for (int iter = 0; iter < args.iters; iter++) {
        uint64_t cycles;
        
        MEASURE_CYCLES(cycles, {
            for (int j = 0; j < NUM_BRANCHES; j++) {
                // Select pattern based on iteration and branch index
                int pattern_idx = ((iter + j) / PATTERN_SIZE) % 4;
                int pos = j % PATTERN_SIZE;
                
                // Conditional branch based on pattern
                if (patterns[pattern_idx][pos]) {
                    branch_counter++;
                } else {
                    branch_counter--;
                }
            }
            COMPILER_BARRIER();
        });
        
        print_csv_row("pht", iter, cycles);
        
        // Periodic flush
        if (iter % 100 == 0) {
            fflush(stdout);
        }
    }
    
    // Use branch_counter to prevent optimization
    if (branch_counter == -999999) {
        printf("# unreachable\n");
    }
    
    return 0;
}
