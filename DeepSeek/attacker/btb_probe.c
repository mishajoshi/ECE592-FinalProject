/**
 * btb_probe.c - Branch Target Buffer (BTB) contention probe
 *
 * Stresses the BTB by executing indirect branches/jumps to many different targets.
 * Uses a function pointer array with pseudo-random selection to maximize BTB pressure.
 *
 * Outputs: CSV with timestamp, probe name, iteration, and cycles per iteration.
 */
#include "common.h"

// BTB probe configuration
#define NUM_TARGETS 512  // Number of different branch targets (exceeds typical BTB capacity)
#define BRANCHES_PER_ITER 256  // Number of indirect branches per iteration

// Simple function targets for indirect branches
static volatile int global_counter = 0;

#define DEFINE_TARGET(n) \
    static void target_##n(void) { \
        global_counter += n; \
        __asm__ __volatile__ ("" ::: "memory"); \
    }

// Generate many target functions
DEFINE_TARGET(0) DEFINE_TARGET(1) DEFINE_TARGET(2) DEFINE_TARGET(3)
DEFINE_TARGET(4) DEFINE_TARGET(5) DEFINE_TARGET(6) DEFINE_TARGET(7)
DEFINE_TARGET(8) DEFINE_TARGET(9) DEFINE_TARGET(10) DEFINE_TARGET(11)
DEFINE_TARGET(12) DEFINE_TARGET(13) DEFINE_TARGET(14) DEFINE_TARGET(15)
DEFINE_TARGET(16) DEFINE_TARGET(17) DEFINE_TARGET(18) DEFINE_TARGET(19)
DEFINE_TARGET(20) DEFINE_TARGET(21) DEFINE_TARGET(22) DEFINE_TARGET(23)
DEFINE_TARGET(24) DEFINE_TARGET(25) DEFINE_TARGET(26) DEFINE_TARGET(27)
DEFINE_TARGET(28) DEFINE_TARGET(29) DEFINE_TARGET(30) DEFINE_TARGET(31)

typedef void (*target_func_t)(void);

// Array of function pointers
static target_func_t targets[NUM_TARGETS];

static void init_targets(void) {
    target_func_t base_targets[32] = {
        target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7,
        target_8, target_9, target_10, target_11, target_12, target_13, target_14, target_15,
        target_16, target_17, target_18, target_19, target_20, target_21, target_22, target_23,
        target_24, target_25, target_26, target_27, target_28, target_29, target_30, target_31
    };
    
    // Fill array by cycling through base targets
    for (int i = 0; i < NUM_TARGETS; i++) {
        targets[i] = base_targets[i % 32];
    }
}

int main(int argc, char** argv) {
    probe_args_t args = parse_probe_args(argc, argv, "btb");
    
    init_targets();
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        for (int j = 0; j < BRANCHES_PER_ITER; j++) {
            int idx = (j * 73) % NUM_TARGETS;  // Pseudo-random target selection
            targets[idx]();
        }
    }
    
    // Print CSV header
    printf(CSV_HEADER);
    fflush(stdout);
    
    // Main measurement loop
    for (int iter = 0; iter < args.iters; iter++) {
        uint64_t cycles;
        
        MEASURE_CYCLES(cycles, {
            for (int j = 0; j < BRANCHES_PER_ITER; j++) {
                // Pseudo-random indirect branch
                int idx = (j * 73 + iter * 5) % NUM_TARGETS;
                targets[idx]();
            }
            COMPILER_BARRIER();
        });
        
        print_csv_row("btb", iter, cycles);
        
        // Periodic flush
        if (iter % 100 == 0) {
            fflush(stdout);
        }
    }
    
    // Use global_counter to prevent optimization
    if (global_counter == -1) {
        printf("# unreachable\n");
    }
    
    return 0;
}
