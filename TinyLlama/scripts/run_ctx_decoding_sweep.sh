#!/usr/bin/env bash
set -euo pipefail

# Context and Decoding sweeps on real llama.cpp victim
# Requires env vars:
#  - VICTIM_BIN: path to llama.cpp CLI binary
#  - MODEL_PATH: path to TinyLLaMA gguf
# Optional:
#  - VCPU, ACPU, REPEATS, SEED, ITERS

VICTIM_BIN=${VICTIM_BIN:-}
MODEL_PATH=${MODEL_PATH:-}
if [[ -z "$VICTIM_BIN" || -z "$MODEL_PATH" ]]; then
  echo "Error: set VICTIM_BIN and MODEL_PATH before running." >&2
  exit 1
fi

VCPU=${VCPU:-2}
ACPU=${ACPU:-6}
REPEATS=${REPEATS:-20}
SEED=${SEED:-42}
ITERS=${ITERS:-2000}

# Ensure probes are built
make -C attacker

echo "=== Starting Context Size Sweep (Probes: cache, tlb) ==="
# Vary Context: 128, 512, 2048. Fixed npredict=64.
CTX_SIZES=(128 512 2048)
N_PREDICT_CTX=64

for probe in cache tlb; do
  for ctx in "${CTX_SIZES[@]}"; do
    for ((r=1; r<=REPEATS; r++)); do
      python3 driver/driver.py \
        --victim-bin "$VICTIM_BIN" \
        --model "$MODEL_PATH" \
        --quant q4_0 --ctx "$ctx" --npredict "$N_PREDICT_CTX" \
        --decoding greedy --temp 0.0 \
        --prompt "The quick brown fox jumps over the lazy dog." --seed "$SEED" --repeat "$r" \
        --victim-cpu "$VCPU" --attacker-cpu "$ACPU" \
        --probe-bin "./attacker/${probe}_probe" --probe "$probe" --iters "$ITERS"
    done
  done
done

echo "=== Starting Decoding Strategy Sweep (Probes: btb, pht) ==="
# Vary Decoding: greedy, sample. Fixed ctx=512, npredict=128.
DECODING_MODES=(greedy sample)
CTX_DEC=512
N_PREDICT_DEC=128

for probe in btb pht; do
  for mode in "${DECODING_MODES[@]}"; do
    # Set temp based on mode
    if [[ "$mode" == "greedy" ]]; then
      TEMP=0.0
    else
      TEMP=1.0
    fi
    
    for ((r=1; r<=REPEATS; r++)); do
      python3 driver/driver.py \
        --victim-bin "$VICTIM_BIN" \
        --model "$MODEL_PATH" \
        --quant q4_0 --ctx "$CTX_DEC" --npredict "$N_PREDICT_DEC" \
        --decoding "$mode" --temp "$TEMP" \
        --prompt "Write a short story about a robot." --seed "$SEED" --repeat "$r" \
        --victim-cpu "$VCPU" --attacker-cpu "$ACPU" \
        --probe-bin "./attacker/${probe}_probe" --probe "$probe" --iters "$ITERS"
    done
  done
done

echo "Sweeps complete. Logs in logs/runs."
echo "Run analysis: python3 analysis/analysis.py --logs-dir logs"
