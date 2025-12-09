#!/usr/bin/env bash
set -euo pipefail

# Sample run using synthetic victim and cache probe
VCPU=${VCPU:-2}
ACPU=${ACPU:-6}
ITERS=${ITERS:-2000}
DIM=${DIM:-512}

# Ensure builds exist
if [[ ! -x ./attacker/cache_probe ]]; then
  make -C attacker
fi
if [[ ! -x ./synthetic/gemm_victim ]]; then
  make -C synthetic
fi

# Start victim in background pinned by driver via taskset
# Note: driver will launch victim itself; this optional line demonstrates standalone victim.
# ./synthetic/gemm_victim "$DIM" 3 &

python3 driver/driver.py \
  --victim-bin ./synthetic/gemm_victim \
  --model ./synthetic/gemm_victim \
  --quant q4_0 --ctx 512 --npredict 64 \
  --decoding greedy --temp 0.0 \
  --prompt "synthetic" --seed 42 --repeat 1 \
  --victim-cpu "$VCPU" --attacker-cpu "$ACPU" \
  --probe-bin ./attacker/cache_probe --probe cache --iters "$ITERS"

# Analyze
python3 analysis/analysis.py --logs-dir logs --output-dir analysis/figs --warmup-discard 500

echo "Sample run and analysis complete. See analysis/figs and logs/runs."