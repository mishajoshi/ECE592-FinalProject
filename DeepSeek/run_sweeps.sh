# #!/bin/bash
# # Example sweep script demonstrating parameter sweeps for SMT contention experiments

# set -e

# # Configuration
# PROJECT_ROOT="/mnt/ncsudrive/m/mjoshi7/FinalProject/microarch_finalproject"
# #VICTIM_BIN="$HOME/llama.cpp/main"
# VICTIM_BIN="$HOME/llama.cpp/build/bin/llama-cli"
# MODEL_DIR="$HOME/llama.cpp/models"
# VICTIM_CPU=0
# ATTACKER_CPU=56  # Must be SMT sibling of VICTIM_CPU

# # Check if victim binary exists 
# if [ ! -f "$VICTIM_BIN" ]; then
#     echo "Error: llama.cpp not found at $VICTIM_BIN"
#     echo "Please install llama.cpp first (see victim/README.md)"
#     exit 1
# fi

# cd "$PROJECT_ROOT"

# # S1: Working Set / Context Sweep
# # echo "=== S1: Working Set / Context Sweep ==="
# # echo "Testing cache and TLB probes with varying context sizes..."

# # for ctx in 128 512 2048; do
# #     for npredict in 16 64 256; do
# #         for repeat in 1 2 3; do
# #             echo "Running: ctx=$ctx, npredict=$npredict, repeat=$repeat (cache)"
# #             python3 driver/driver.py \
# #                 --root . \
# #                 --victim-bin "$VICTIM_BIN" \
# #                 --model "$MODEL_DIR/tinyllama-1.1b-q4_0.gguf" \
# #                 --quant q4_0 \
# #                 --ctx $ctx \
# #                 --npredict $npredict \
# #                 --decoding greedy \
# #                 --victim-cpu $VICTIM_CPU \
# #                 --attacker-cpu $ATTACKER_CPU \
# #                 --probe-bin ./attacker/cache_probe \
# #                 --probe cache \
# #                 --iters 2000 \
# #                 --repeat $repeat || echo "Run failed, continuing..."
            
# #             sleep 2  # Cool-down between runs
# #         done
# #     done
# # done

# # # S2: Quantization Width Sweep
# # echo
# # echo "=== S2: Quantization Width Sweep ==="
# # echo "Testing different quantization levels..."

# # for quant in q4_0 q5_0 q8_0; do
# #     model_file="$MODEL_DIR/tinyllama-1.1b-$quant.gguf"
    
# #     if [ ! -f "$model_file" ]; then
# #         echo "Warning: Model $model_file not found, skipping..."
# #         continue
# #     fi
    
# #     for repeat in 1 2 3; do
# #         for probe_name in cache tlb; do
# #             echo "Running: quant=$quant, repeat=$repeat, probe=$probe_name"
# #             python3 driver/driver.py \
# #                 --root . \
# #                 --victim-bin "$VICTIM_BIN" \
# #                 --model "$model_file" \
# #                 --quant $quant \
# #                 --ctx 512 \
# #                 --npredict 64 \
# #                 --decoding greedy \
# #                 --victim-cpu $VICTIM_CPU \
# #                 --attacker-cpu $ATTACKER_CPU \
# #                 --probe-bin ./attacker/${probe_name}_probe \
# #                 --probe $probe_name \
# #                 --iters 2000 \
# #                 --repeat $repeat || echo "Run failed, continuing..."
            
# #             sleep 2
# #         done
# #     done
# # done

# # # S3: Access Pattern / Decoding Sweep
# # echo
# # echo "=== S3: Access Pattern / Decoding Sweep ==="
# # echo "Testing greedy vs sampling decoding with branch predictor probes..."

# # for decoding in greedy sample; do
# #     for repeat in 1 2 3; do
# #         for probe_name in btb pht; do
# #             echo "Running: decoding=$decoding, repeat=$repeat, probe=$probe_name"
            
# #             if [ "$decoding" == "greedy" ]; then
# #                 python3 driver/driver.py \
# #                     --root . \
# #                     --victim-bin "$VICTIM_BIN" \
# #                     --model "$MODEL_DIR/tinyllama-1.1b-q4_0.gguf" \
# #                     --quant q4_0 \
# #                     --ctx 512 \
# #                     --npredict 64 \
# #                     --decoding greedy \
# #                     --victim-cpu $VICTIM_CPU \
# #                     --attacker-cpu $ATTACKER_CPU \
# #                     --probe-bin ./attacker/${probe_name}_probe \
# #                     --probe $probe_name \
# #                     --iters 2000 \
# #                     --repeat $repeat || echo "Run failed, continuing..."
# #             else
# #                 python3 driver/driver.py \
# #                     --root . \
# #                     --victim-bin "$VICTIM_BIN" \
# #                     --model "$MODEL_DIR/tinyllama-1.1b-q4_0.gguf" \
# #                     --quant q4_0 \
# #                     --ctx 512 \
# #                     --npredict 64 \
# #                     --decoding sample \
# #                     --temp 1.0 \
# #                     --top-k 40 \
# #                     --top-p 0.95 \
# #                     --victim-cpu $VICTIM_CPU \
# #                     --attacker-cpu $ATTACKER_CPU \
# #                     --probe-bin ./attacker/${probe_name}_probe \
# #                     --probe $probe_name \
# #                     --iters 2000 \
# #                     --repeat $repeat || echo "Run failed, continuing..."
# #             fi
            
# #             sleep 2
# #         done
# #     done
# # done

# #npredict sweep
# # for np in 16 64 256; do
# #   for rep in 1 2 3; do
# #     python3 driver/driver.py \
# #       --root . \
# #       --victim-bin ~/llama.cpp/build/bin/llama-cli \
# #       --model ~/llama.cpp/models/tinyllama-1.1b-q4_0.gguf \
# #       --quant q4_0 \
# #       --ctx 512 \
# #       --npredict $np \
# #       --decoding greedy \
# #       --victim-cpu 0 \
# #       --attacker-cpu 56 \
# #       --probe-bin ./attacker/cache_probe \
# #       --probe cache \
# #       --iters 2000 \
# #       --repeat $rep
# #   done
# # done

# # S4: Temperature Sweep
# # TEMPS=(0.0 0.5 0.8)
# # REPEATS=(1 2 3)

# # for t in "${TEMPS[@]}"; do
# #   for rep in "${REPEATS[@]}"; do

# #     echo "Running temp = $t   repeat = $rep"

# #     python3 driver/driver.py \
# #       --root . \
# #       --victim-bin "$VICTIM_BIN" \
# #       --model "$MODEL" \
# #       --quant q4_0 \
# #       --ctx 512 \
# #       --npredict 64 \
# #       --decoding sample \
# #       --temp $t \
# #       --victim-cpu $VICTIM_CPU \
# #       --attacker-cpu $ATTACKER_CPU \
# #       --probe-bin ./attacker/pht_probe \
# #       --probe pht \
# #       --iters 2000 \
# #       --repeat $rep

# #   done
# # done

# # S4: Prompt Semantics Sweep
# echo
# echo "=== S4: Prompt Semantics Sweep ==="
# echo "Testing prompt semantic leakage..."

# declare -A PROMPTS=(
#     ["nl"]="Write a detailed 5-sentence story about a traveler who discovers an abandoned city under the ocean. Describe imagery."
#     ["math"]="Compute the integral ∫ 0→∞ x^3 * e^(−x^2/2) dx and show each substitution step clearly. Write using long derivations."
#     ["code"]="Write a Python implementation of Dijkstra’s shortest-path algorithm with adjacency lists, min-heap, and complexity analysis."
# )

# for label in "${!PROMPTS[@]}"; do
#     prompt="${PROMPTS[$label]}"

#     for repeat in 1 2 3; do
#         echo "Running: prompt_label=$label, repeat=$repeat"

#         python3 driver/driver.py \
#             --root . \
#             --victim-bin "$VICTIM_BIN" \
#             --model "$MODEL_DIR/tinyllama-1.1b-q4_0.gguf" \
#             --quant q4_0 \
#             --ctx 512 \
#             --npredict 128 \
#             --decoding greedy \
#             --prompt "$prompt" \
#             --prompt-label "$label" \
#             --victim-cpu $VICTIM_CPU \
#             --attacker-cpu $ATTACKER_CPU \
#             --probe-bin ./attacker/cache_probe \
#             --probe btb \
#             --iters 20000 \
#             --repeat $repeat || echo "Run failed, continuing..."

#         sleep 2
#     done
# done






# echo
# echo "=== S-DeepSeek: Basic Sweep for DeepSeek 8B ==="
# echo "Running DeepSeek-R1-Distill-Llama-8B model"

# DEEPSEEK_MODEL="$HOME/llama.cpp/models/deepseek-llama8b-q4.gguf"

# # for ctx in 128 512; do
# #     for decoding in greedy sample; do
# #         for repeat in 1 2; do
# #             echo "DeepSeek: ctx=$ctx, decoding=$decoding, repeat=$repeat"

# #             python3 driver/driver.py \
# #                 --root . \
# #                 --victim-bin "$VICTIM_BIN" \
# #                 --model "$DEEPSEEK_MODEL" \
# #                 --quant "q4" \
# #                 --ctx $ctx \
# #                 --npredict 64 \
# #                 --decoding $decoding \
# #                 --victim-cpu $VICTIM_CPU \
# #                 --attacker-cpu $ATTACKER_CPU \
# #                 --probe-bin ./attacker/btb_probe \
# #                 --probe btb \
# #                 --iters 2000 \
# #                 --repeat $repeat || echo "DeepSeek run failed"
# #             sleep 2
# #         done
# #     done
# # done


# # S4: Prompt Semantics Sweep
# echo
# echo "=== S4: Prompt Semantics Sweep ==="
# echo "Testing prompt semantic leakage..."

# declare -A PROMPTS=(
#     ["nl"]="Write a detailed 5-sentence story about a traveler who discovers an abandoned city under the ocean. Describe imagery."
#     ["math"]="Compute the integral ∫ 0→∞ x^3 * e^(−x^2/2) dx and show each substitution step clearly. Write using long derivations."
#     ["code"]="Write a Python implementation of Dijkstra’s shortest-path algorithm with adjacency lists, min-heap, and complexity analysis."
# )

# for label in "${!PROMPTS[@]}"; do
#     prompt="${PROMPTS[$label]}"

#     for repeat in 1 2 3; do
#         echo "Running: prompt_label=$label, repeat=$repeat"

#         python3 driver/driver.py \
#             --root . \
#             --victim-bin "$VICTIM_BIN" \
#             --model "$MODEL_DIR/tinyllama-1.1b-q4_0.gguf" \
#             --quant q4_0 \
#             --ctx 512 \
#             --npredict 128 \
#             --decoding greedy \
#             --prompt "$prompt" \
#             --prompt-label "$label" \
#             --victim-cpu $VICTIM_CPU \
#             --attacker-cpu $ATTACKER_CPU \
#             --probe-bin ./attacker/cache_probe \
#             --probe btb \
#             --iters 20000 \
#             --repeat $repeat || echo "Run failed, continuing..."

#         sleep 2
#     done
# done

# echo
# echo "=== All sweeps complete! ==="
# echo "Run analysis with: python3 analysis/analysis.py --logs-dir logs"


#------------------------------------------------------------------------------------
#!/bin/bash
# Unified Sweep Script for DeepSeek 8B SMT Leakage Experiments
set -e

###########################################
# Configuration
###########################################

PROJECT_ROOT="/mnt/ncsudrive/m/mjoshi7/FinalProject/microarch_finalproject"
VICTIM_BIN="$HOME/llama.cpp/build/bin/llama-cli"
MODEL="$HOME/llama.cpp/models/deepseek-llama8b-q4.gguf"

VICTIM_CPU=0
ATTACKER_CPU=56   # SMT sibling

###########################################
# Startup Checks
###########################################

if [ ! -f "$VICTIM_BIN" ]; then
    echo "ERROR: llama-cli not found at $VICTIM_BIN"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: DeepSeek model not found at $MODEL"
    exit 1
fi

cd "$PROJECT_ROOT"

echo
echo "==============================================="
echo "   DeepSeek 8B — Full SMT Leakage Experiment"
echo "==============================================="
echo "Model: $MODEL"
echo "Victim CPU: $VICTIM_CPU"
echo "Attacker CPU: $ATTACKER_CPU"
echo

############################################################
# S1 — Context Size Sweep
############################################################
echo
echo "=== S1: Context Size / Working-Set Sweep ==="

for ctx in 128 512 2048; do
  for np in 16 64 256; do
    for rep in 1 2 3; do

      echo "CTX=$ctx NPREDICT=$np Repeat=$rep"

      python3 driver/driver.py \
        --root . \
        --victim-bin "$VICTIM_BIN" \
        --model "$MODEL" \
        --quant q4 \
        --ctx $ctx \
        --npredict $np \
        --decoding greedy \
        --victim-cpu $VICTIM_CPU \
        --attacker-cpu $ATTACKER_CPU \
        --probe-bin ./attacker/cache_probe \
        --probe cache \
        --iters 2000 \
        --repeat $rep || echo "Failed run (ctx), continuing..."

      sleep 1
    done
  done
done


############################################################
# S2 — Decoding Strategy (Greedy vs Sampling)
############################################################
echo
echo "=== S2: Decoding Strategy Sweep ==="

for decoding in greedy sample; do
  for rep in 1 2 3; do
    for probe in btb pht; do

      echo "Decoding=$decoding Probe=$probe Repeat=$rep"

      if [ "$decoding" == "greedy" ]; then
        python3 driver/driver.py \
          --root . \
          --victim-bin "$VICTIM_BIN" \
          --model "$MODEL" \
          --quant q4 \
          --ctx 512 \
          --npredict 100 \
          --decoding greedy \
          --victim-cpu $VICTIM_CPU \
          --attacker-cpu $ATTACKER_CPU \
          --probe-bin ./attacker/${probe}_probe \
          --probe $probe \
          --iters 2000 \
          --repeat $rep
      else
        python3 driver/driver.py \
          --root . \
          --victim-bin "$VICTIM_BIN" \
          --model "$MODEL" \
          --quant q4 \
          --ctx 512 \
          --npredict 100 \
          --decoding sample \
          --temp 1.0 \
          --top-k 40 \
          --top-p 0.95 \
          --victim-cpu $VICTIM_CPU \
          --attacker-cpu $ATTACKER_CPU \
          --probe-bin ./attacker/${probe}_probe \
          --probe $probe \
          --iters 2000 \
          --repeat $rep
      fi

      sleep 1
    done
  done
done


############################################################
# S3 — Temperature Sweep
############################################################
echo
echo "=== S3: Temperature Sweep ==="

for temp in 0.0 0.5 1.3; do
  for rep in 1 2 3; do

    echo "Temp=$temp Repeat=$rep"

    python3 driver/driver.py \
      --root . \
      --victim-bin "$VICTIM_BIN" \
      --model "$MODEL" \
      --quant q4 \
      --ctx 512 \
      --npredict 100 \
      --decoding sample \
      --temp $temp \
      --top-k 40 \
      --top-p 0.95 \
      --victim-cpu $VICTIM_CPU \
      --attacker-cpu $ATTACKER_CPU \
      --probe-bin ./attacker/pht_probe \
      --probe pht \
      --iters 2000 \
      --repeat $rep || echo "Temperature sweep failed"

    sleep 1
  done
done


############################################################
# S4 — Prompt Semantics Sweep (High Accuracy)
############################################################
echo
echo "=== S4: Prompt Semantics Sweep ==="
echo "DeepSeek 8B — Prompt Category Leakage"

REPEATS=10   # change to 25 later when accuracy optimized

declare -A PROMPTS=(
  ["nl"]="Write a detailed vivid emotional 200-word story about an explorer in a distorted time forest."
  ["math"]="Derive ∫₀∞ x⁴ e^(−x²/3) dx and generalize to ∫₀∞ xⁿ e^(−a x²) dx using gamma functions with full symbolic steps."
  ["code"]="Write a full multi-threaded C++ task scheduler using work-stealing deques, condition variables, and error handling."
)

for label in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$label]}"

  for rep in $(seq 1 $REPEATS); do
    for probe in pht btb; do
      echo "Prompt=$label Probe=$probe Repeat=$rep"

      python3 driver/driver.py \
        --root . \
        --victim-bin "$VICTIM_BIN" \
        --model "$MODEL" \
        --quant q4 \
        --ctx 512 \
        --npredict 800 \
        --decoding sample \
        --temp 0.9 \
        --top-k 40 \
        --prompt "$prompt" \
        --prompt-label "$label" \
        --victim-cpu $VICTIM_CPU \
        --attacker-cpu $ATTACKER_CPU \
        --probe-bin ./attacker/${probe}_probe \
        --probe $probe \
        --iters 20000 \
        --repeat $rep || echo "Prompt run failed"

      sleep 2
    done
  done
done


echo
echo "=== ALL DeepSeek Sweeps Complete ==="
echo "Run analysis with:"
echo "  python3 analysis/analysis.py --logs-dir logs"
echo "====================================="
