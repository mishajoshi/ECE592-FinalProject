#!/usr/bin/env bash
set -euo pipefail

# BTB-only prompt semantics sweep on real llama.cpp victim
# Requires env vars:
#  - VICTIM_BIN: path to llama.cpp CLI binary (e.g., ~/llama.cpp/llama-cli or ./main)
#  - MODEL_PATH: path to TinyLLaMA gguf (e.g., ~/llama.cpp/models/tinyllama-1.1b-q4_0.gguf)
# Optional env vars with sane defaults:
#  - VCPU: victim CPU id
#  - ACPU: attacker CPU id (SMT sibling of VCPU)
#  - REPEATS: repeats per class (default 20)
#  - N_PREDICT: tokens to generate (default 512)
#  - CTX: context size (default 512)
#  - SEED: random seed (default 42)
#  - ITERS: probe iterations (default 2000)

VICTIM_BIN=${VICTIM_BIN:-}
MODEL_PATH=${MODEL_PATH:-}
if [[ -z "$VICTIM_BIN" || -z "$MODEL_PATH" ]]; then
  echo "Error: set VICTIM_BIN and MODEL_PATH before running." >&2
  echo "Example: export VICTIM_BIN=~/llama.cpp/llama-cli; export MODEL_PATH=~/llama.cpp/models/tinyllama-1.1b-q4_0.gguf" >&2
  exit 1
fi

VCPU=${VCPU:-2}
ACPU=${ACPU:-6}
REPEATS=${REPEATS:-20}
N_PREDICT=${N_PREDICT:-512}
CTX=${CTX:-512}
SEED=${SEED:-42}
ITERS=${ITERS:-2000}

# Ensure BTB probe is built
if [[ ! -x ./attacker/btb_probe ]]; then
  make -C attacker btb_probe
fi

# Stricter prompt templates
PROMPT_MATH=$(cat <<'EOF'
You are a precise math assistant.
Solve the problem step by step with numbered equations and final answer.
Problem: Compute the sum S = sum_{k=1}^{100} k^2. Provide derivation using known formulas and the final numeric value.
EOF
)

PROMPT_CODE=$(cat <<'EOF'
You are a programming assistant.
Write a Python function with loops and conditionals that computes factorial(n) iteratively, handles edge cases, and includes a docstring.
Return only the function code; no extra commentary.
EOF
)

PROMPT_NL=$(cat <<'EOF'
You are a creative writer.
Compose a narrative paragraph describing a sunrise over the ocean, focusing on imagery and sensory details without using lists or code.
EOF
)

declare -A PROMPTS
PROMPTS=(
  [math]="$PROMPT_MATH"
  [code]="$PROMPT_CODE"
  [nl]="$PROMPT_NL"
)

echo "Starting BTB-only prompt sweep: repeats=$REPEATS npredict=$N_PREDICT ctx=$CTX"

for label in math code nl; do
  prompt=${PROMPTS[$label]}
  for ((r=1; r<=REPEATS; r++)); do
    python3 driver/driver.py \
      --victim-bin "$VICTIM_BIN" \
      --model "$MODEL_PATH" \
      --quant q4_0 --ctx "$CTX" --npredict "$N_PREDICT" \
      --decoding greedy --temp 0.0 \
      --prompt "$prompt" --seed "$SEED" --repeat "$r" \
      --victim-cpu "$VCPU" --attacker-cpu "$ACPU" \
      --probe-bin ./attacker/btb_probe --probe btb --iters "$ITERS" \
      --prompt-label "$label"
  done
done

echo "BTB-only sweep complete. Logs in logs/runs."
echo "Run analysis: python3 analysis/analysis.py --logs-dir logs"
