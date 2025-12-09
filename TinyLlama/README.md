# TinyLLaMA Side-Channel Attack Experiments

This project implements microarchitectural side-channel attacks on TinyLLaMA-1.1B inference running on CPU via llama.cpp. The attack setup consists of a victim process running LLM inference on one CPU core and attacker probes running on its SMT (Simultaneous Multi-Threading) sibling, measuring timing contention to infer inference parameters.

## Overview

**Attack Targets:**
- **Context Size**: Distinguish between 128, 512, and 2048 token contexts (3-class classification)
- **Decoding Strategy**: Distinguish between greedy and sampling decoding (2-class classification)  
- **Prompt Semantics**: Classify prompt types - Math, Code, Natural Language, Custom (4-class classification)

**Probe Types:**
- **Cache Probe**: Measures LLC (Last Level Cache) contention via Prime+Probe
- **TLB Probe**: Measures Translation Lookaside Buffer contention
- **BTB Probe**: Measures Branch Target Buffer contention
- **PHT Probe**: Measures Pattern History Table contention

## Project Structure
- `driver/`: Orchestration (`driver.py`) and utilities (`run_utils.py`)
- `attacker/`: C implementations of microarchitectural probes with `Makefile`
- `synthetic/`: Synthetic GEMM victim for testing (optional)
- `analysis/`: Statistical analysis and ML classifier training (`analysis.py`)
- `logs/`: Runtime outputs organized as `runs/<run_id>/` with measurements and metadata
- `scripts/`: Automation scripts for running sweeps

## Prerequisites

**System Requirements:**
- Linux system with SMT-enabled CPU (Intel Xeon tested)
- `taskset` for CPU pinning
- GCC compiler (for building probe binaries)
- `perf` (optional, for additional profiling)

**Software Requirements:**
- Python 3.9 or later
- Python packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- **llama.cpp**: Compiled binary (`main`) for running TinyLLaMA inference
- **Model**: TinyLLaMA-1.1B GGUF format (Q4_0 quantization recommended, ~600MB)

**Install Python Dependencies:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

**Get llama.cpp and TinyLLaMA Model:**

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
# Binary will be at ./main

# Download TinyLLaMA GGUF model (example)
# Get from Hugging Face: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Place model file somewhere accessible, e.g., ~/models/tinyllama-1.1b-q4_0.gguf
```

## Build Attack Probes

Build all microarchitectural probe binaries:

```bash
make -C attacker
```

This creates:
- `attacker/cache_probe` - Cache contention probe
- `attacker/tlb_probe` - TLB contention probe  
- `attacker/btb_probe` - Branch Target Buffer probe
- `attacker/pht_probe` - Pattern History Table probe

Alternatively, use the build script:

```bash
bash scripts/build_all.sh
```

## Identify SMT Sibling CPUs

SMT siblings share physical core resources, enabling side-channel attacks. Find sibling CPU pairs:

```bash
python3 -c "from driver.run_utils import detect_siblings; print(detect_siblings())"
```

**Example output:**
```
{0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}
```

This means CPUs 2 and 6 are SMT siblings. Pick a pair for victim and attacker:
- `--victim-cpu 2` (runs LLM inference)
- `--attacker-cpu 6` (runs probe, measuring contention from CPU 2)

## Running Experiments

### Single Run Example

Run a single experiment attacking context size inference with cache probe:

```bash
# Set environment variables
export VICTIM_BIN=/path/to/llama.cpp/main
export MODEL_PATH=/path/to/tinyllama-1.1b-q4_0.gguf

# Run single experiment
python3 driver/driver.py \
  --victim-bin "$VICTIM_BIN" \
  --model "$MODEL_PATH" \
  --quant q4_0 \
  --ctx 512 \
  --npredict 64 \
  --decoding greedy \
  --temp 0.0 \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --seed 42 \
  --repeat 1 \
  --victim-cpu 2 \
  --attacker-cpu 6 \
  --probe-bin ./attacker/cache_probe \
  --probe cache \
  --iters 2000
```

**Key Parameters:**
- `--ctx`: Context window size (128, 512, or 2048 tokens)
- `--npredict`: Number of tokens to generate (64, 128, 256, or 512)
- `--decoding`: Strategy - `greedy` (deterministic) or `sample` (stochastic)
- `--temp`: Temperature for sampling (0.0 for greedy, 1.0 for sampling)
- `--probe`: Which probe to use (`cache`, `tlb`, `btb`, or `pht`)
- `--iters`: Number of probe iterations (default 2000)
- `--repeat`: Repeat index for multiple runs of same config

**Output Location:**
Each run creates a directory: `logs/runs/<run_id>/` containing:
- `attacker_stdout.txt`: Probe measurements in CSV format (`ts_ns,probe,iter,cycles`)
- `victim_stdout.txt`: LLM inference logs
- `freq.csv`: CPU frequency samples during execution
- `meta.json`: Full experiment metadata (config, timings, host info)
- `timings.json`: Execution timing breakdown
- Master index: `logs/index.csv` tracks all runs

### Run Complete Sweeps

**Context Size Attack (Cache + TLB probes):**
```bash
export VICTIM_BIN=/path/to/llama.cpp/main
export MODEL_PATH=/path/to/tinyllama-1.1b-q4_0.gguf
export VCPU=2
export ACPU=6
export REPEATS=20  # Run each config 20 times for statistical robustness

bash scripts/run_ctx_decoding_sweep.sh
```

This sweeps:
- Context sizes: 128, 512, 2048 tokens
- Probes: cache, tlb
- Decoding: greedy (fixed for context experiments)
- Repeats: 20 runs per configuration
- **Total runs**: 3 contexts × 2 probes × 20 repeats = 120 runs

**Decoding Strategy Attack (BTB + PHT probes):**
The same script also runs decoding sweeps:
- Decoding modes: greedy, sampling
- Context: 512 tokens (fixed)
- Probes: btb, pht
- Repeats: 20 runs per configuration
- **Total runs**: 2 decoding modes × 2 probes × 20 repeats = 80 runs

**Prompt Semantics Attack (BTB + PHT probes):**
```bash
export VICTIM_BIN=/path/to/llama.cpp/main
export MODEL_PATH=/path/to/tinyllama-1.1b-q4_0.gguf
export VCPU=2
export ACPU=6
export REPEATS=20

bash scripts/run_btb_prompt_sweep.sh
```

This sweeps:
- Prompt types: Math, Code, Natural Language, Custom
- Probes: btb, pht
- Context: 512 tokens (fixed)
- Repeats: 20 runs per configuration
- **Total runs**: 4 prompt types × 2 probes × 20 repeats = 160 runs

**Expected Runtime:**
- Single run: ~10-30 seconds (depends on generation length)
- Full context sweep: ~30-60 minutes
- Full decoding sweep: ~20-40 minutes
- Full semantics sweep: ~40-80 minutes

## Analysis and Results

After collecting experimental data, run the analysis pipeline to compute statistics, train classifiers, and generate visualizations:

```bash
python3 analysis/analysis.py \
  --logs-dir logs \
  --output-dir analysis/figs \
  --warmup-discard 500
```

**Parameters:**
- `--logs-dir`: Directory containing run data (default: `logs`)
- `--output-dir`: Where to save output figures (default: `analysis/figs`)
- `--warmup-discard`: Number of initial probe iterations to discard (default: 500)

### Analysis Pipeline

The script performs:
1. **Data Loading**: Reads all runs from `logs/runs/` with metadata
2. **Statistical Feature Extraction**: Computes per-run features:
   - Mean, median, standard deviation
   - Percentiles: p10, p50, p90, p99
   - Skewness and kurtosis (distribution shape)
3. **Classification**: Trains Random Forest classifiers to distinguish:
   - Context sizes (128 vs 512 vs 2048)
   - Decoding strategies (greedy vs sampling)
   - Prompt semantics (Math vs Code vs NL vs Custom)
4. **Visualization**: Generates comprehensive figures

### Generated Outputs

**Statistical Summary:**
- `run_statistics.csv`: Per-run aggregate statistics (mean, std, skewness, kurtosis)

**Classification Results:**
- `confusion_matrix_context.png`: 3×3 confusion matrix for context size classification
- `confusion_matrix_decoding.png`: 2×2 confusion matrix for decoding strategy
- `confusion_matrix_semantics.png`: 4×4 confusion matrix for prompt semantics
- `feature_importance_context.png`: Top features for context classification
- `feature_importance_decoding.png`: Top features for decoding classification
- `feature_importance_semantics.png`: Top features for semantic classification

**Visualizations:**
- `hist_cycles_cache.png`: Histogram of cache probe cycle counts
- `hist_cycles_tlb.png`: Histogram of TLB probe cycle counts
- `hist_cycles_btb.png`: Histogram of BTB probe cycle counts
- `hist_cycles_pht.png`: Histogram of PHT probe cycle counts
- `boxplot_context.png`: Box plots comparing cycle distributions across context sizes
- `boxplot_decoding.png`: Box plots comparing greedy vs sampling
- `pca_runs.png`: 2D PCA projection showing cluster separation
- `skew_kurtosis_analysis.png`: Distribution shape analysis

### Expected Results (TinyLLaMA-1.1B)

Based on experimental evaluation:

**Context Size Classification:**
- **Accuracy**: ~80.5% (3-class: 128/512/2048 tokens)
- **Best Probe**: Cache (LLC contention strongly correlates with context size)
- **F1 Scores**: 128-token: 0.84, 512-token: 0.78, 2048-token: 0.81
- **Key Features**: Kurtosis, p99, standard deviation

**Decoding Strategy Classification:**
- **Accuracy**: ~90.0% (2-class: greedy/sampling)
- **Best Probe**: BTB (branch prediction patterns differ significantly)
- **F1 Scores**: Greedy: 0.92, Sampling: 0.87
- **Key Features**: Skewness, median, kurtosis
- **Asymmetry**: Greedy easier to detect (95% recall) than sampling (82% recall)

**Prompt Semantics Classification:**
- **Accuracy**: ~61.4% (4-class: Math/Code/NL/Custom)
- **Best Probe**: PHT (control flow patterns vary by content type)
- **F1 Scores**: Math: 0.42, Code: 0.44, NL: 0.47, Custom: 0.95
- **Observation**: Custom prompts highly distinguishable (93% recall), generic categories harder

**Distribution Characteristics:**
- **Skewness**: 21.7 (highly right-skewed timing distributions)
- **Kurtosis**: 666 (extreme outliers present, heavy-tailed distributions)
- **Cycle Range**: 200-50,000+ cycles depending on configuration

## Understanding the Attack

### Why Side Channels Work

**Context Size Leakage:**
- Larger contexts → more memory accesses → higher cache/TLB contention
- KV-cache size scales linearly with context length
- Probe observes cache eviction patterns correlating with context

**Decoding Strategy Leakage:**
- Greedy: Deterministic, predictable branches → consistent BTB patterns
- Sampling: Stochastic, random sampling → variable BTB/PHT behavior
- Temperature parameter affects branch predictability

**Semantic Leakage:**
- Different content → different token distributions → varied computation patterns
- Math: Numeric tokens, arithmetic operations
- Code: Structured syntax, repetitive patterns
- Custom prompts: Out-of-distribution vocabulary creates distinctive signatures

### Machine Learning Approach

**Classifier**: Random Forest (100 trees)
- Ensemble method robust to noise
- Handles high-dimensional feature space
- Provides feature importance rankings

**Features**: 8 per run (computed from ~1500-2000 cycle measurements)
- Central tendency: mean, median
- Spread: std, p10, p90, p99
- Shape: skewness (asymmetry), kurtosis (tail weight)

**Training**: 80/20 train-test split with stratification
- Cross-validation for hyperparameter tuning
- Balanced classes via stratified sampling

## Important Notes

**System Considerations:**
- **SMT Required**: Attack only works on SMT siblings sharing physical core
- **CPU Pinning**: `taskset` ensures victim/attacker on correct cores
- **Noise**: Real systems have background noise; use multiple repeats
- **Warmup**: First ~500 probe iterations discarded (warmup effects)
- **Frequency Scaling**: CPU frequency sampled to detect DVFS interference

**Security Implications:**
- Unprivileged attacker (no special permissions required)
- Co-location sufficient (cloud multi-tenancy scenarios)
- Software-only mitigation difficult (architectural issue)
- Defenses: Disable SMT, context padding, decoding normalization

**Limitations:**
- Controlled experimental setup (isolated system)
- Fixed generation lengths (variable length adds complexity)
- Single model architecture (TinyLLaMA specific)
- Limited prompt diversity (4 semantic categories)

## Troubleshooting

**"Cannot detect SMT siblings":**
- Verify SMT enabled in BIOS
- Check `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`

**"Permission denied" on CPU pinning:**
- Run with sufficient privileges or adjust `taskset` permissions
- Some systems restrict CPU affinity for non-root users

**Low classification accuracy:**
- Ensure sufficient repeats (20+ per config)
- Check for system noise (disable turbo boost, close background apps)
- Verify victim/attacker on true SMT siblings

**Probe crashes or hangs:**
- Reduce `--iters` if memory constrained
- Check probe compilation (`make clean && make -C attacker`)
- Verify CPU IDs are valid

## Quick Reference

**Setup:**
```bash
# Build probes
make -C attacker

# Find SMT siblings
python3 -c "from driver.run_utils import detect_siblings; print(detect_siblings())"

# Set environment
export VICTIM_BIN=/path/to/llama.cpp/main
export MODEL_PATH=/path/to/tinyllama-1.1b-q4_0.gguf
export VCPU=2 ACPU=6
```

**Run Experiments:**
```bash
# Single run (context size)
python3 driver/driver.py --victim-bin "$VICTIM_BIN" --model "$MODEL_PATH" \
  --ctx 512 --npredict 64 --decoding greedy --prompt "Test prompt" \
  --victim-cpu $VCPU --attacker-cpu $ACPU \
  --probe-bin ./attacker/cache_probe --probe cache --iters 2000 --seed 42 --repeat 1

# Full sweep (all experiments)
bash scripts/run_ctx_decoding_sweep.sh    # Context + Decoding
bash scripts/run_btb_prompt_sweep.sh      # Semantics
```

**Analyze Results:**
```bash
python3 analysis/analysis.py --logs-dir logs --output-dir analysis/figs
```

**Check Results:**
```bash
# View run index
head -5 logs/index.csv

# Count runs by probe type
cut -d',' -f7 logs/index.csv | sort | uniq -c

# View sample probe data
head -10 logs/runs/*/attacker_stdout.txt | head -20
```
```
