# DeepSeek-8B Side-Channel Attack Experiments

This project implements microarchitectural side-channel attacks on DeepSeek-R1-Distill-Llama-8B inference running on CPU via llama.cpp. The attack setup consists of a victim process running LLM inference on one CPU core and attacker probes running on its SMT (Simultaneous Multi-Threading) sibling, measuring timing contention to infer inference parameters.

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

**Key Findings:**
DeepSeek-8B (8 billion parameters) shows **amplified leakage** compared to TinyLLaMA-1.1B:
- **2.2× higher kurtosis** (1498 vs 666) - more extreme timing outliers
- **1.8× higher skewness** (38.7 vs 21.7) - stronger distribution asymmetry
- **Better classification accuracy** across all attack types due to stronger signals

## Project Structure
- `driver/`: Orchestration (`driver.py`) and utilities (`run_utils.py`)
- `attacker/`: C implementations of microarchitectural probes with `Makefile`
- `victim/`: Documentation for llama.cpp setup
- `analysis/`: Statistical analysis and ML classifier training (`analysis.py`)
- `logs/`: Runtime outputs organized as `runs/<run_id>/` with measurements and metadata
- `run_sweeps.sh`: Master automation script for running all experiments

## Prerequisites

**System Requirements:**
- Linux system with SMT-enabled CPU (Intel Xeon tested)
- `taskset` for CPU pinning
- Build tools: `gcc`, `make`, `cmake`, `git`
- `perf` (optional, for additional profiling)

**Software Requirements:**
- Python 3.9 or later
- Python packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- **llama.cpp**: Compiled binary (`llama-cli`) for running DeepSeek inference
- **Model**: DeepSeek-R1-Distill-Llama-8B GGUF format (Q4_K_M quantization recommended, ~5GB)

**Install Python Dependencies:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

**Get llama.cpp and DeepSeek Model:**

```bash
# Clone and build llama.cpp (with optimized build)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
# Binary will be at build/bin/llama-cli

# Optional: Enable BLAS for faster CPU inference
# cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

# Download DeepSeek-R1-Distill-Llama-8B GGUF model (Q4_K_M, ~5GB)
mkdir -p models
cd models
wget https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/deepseek-r1-distill-llama-8b.Q4_K_M.gguf \
  -O deepseek-llama8b-q4.gguf

# Or use huggingface-cli for faster download
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF \
  deepseek-r1-distill-llama-8b.Q4_K_M.gguf --local-dir models
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

## Identify SMT Sibling CPUs

SMT siblings share physical core resources, enabling side-channel attacks. Find sibling CPU pairs:

```bash
python3 -c "from driver.run_utils import detect_siblings; print(detect_siblings())"
```

**Example output:**
```
{0: [0, 56], 1: [1, 57], 2: [2, 58], 3: [3, 59]}
```

This means CPUs 0 and 56 are SMT siblings. Pick a pair for victim and attacker:
- `--victim-cpu 0` (runs LLM inference)
- `--attacker-cpu 56` (runs probe, measuring contention from CPU 0)

**Alternative method:**
```bash
grep . /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
```

## Running Experiments

### Single Run Example

Run a single experiment attacking context size inference with cache probe:

```bash
# Set paths (adjust to your environment)
export VICTIM_BIN=~/llama.cpp/build/bin/llama-cli
export MODEL_PATH=~/llama.cpp/models/deepseek-llama8b-q4.gguf

# Run single experiment
python3 driver/driver.py \
  --victim-bin "$VICTIM_BIN" \
  --model "$MODEL_PATH" \
  --quant q4_k_m \
  --ctx 512 \
  --npredict 64 \
  --decoding greedy \
  --temp 0.0 \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --seed 42 \
  --repeat 1 \
  --victim-cpu 0 \
  --attacker-cpu 56 \
  --probe-bin ./attacker/cache_probe \
  --probe cache \
  --iters 2000
```

**Key Parameters:**
- `--ctx`: Context window size (128, 512, or 2048 tokens)
- `--npredict`: Number of tokens to generate (16, 64, 128, 256, or 512)
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

**Configure Sweep Script:**

Edit `run_sweeps.sh` and set these variables:
```bash
PROJECT_ROOT="/path/to/ECE592-FinalProject/DeepSeek"
VICTIM_BIN="$HOME/llama.cpp/build/bin/llama-cli"
MODEL_DIR="$HOME/llama.cpp/models"
VICTIM_CPU=0
ATTACKER_CPU=56  # Must be SMT sibling of VICTIM_CPU
```

**Run Full Experiment Suite:**
```bash
bash run_sweeps.sh
```

This automated script runs:

1. **Context Size Attack** (Cache + TLB probes)
   - Context sizes: 128, 512, 2048 tokens
   - Generation lengths: 16, 64, 256 tokens
   - Probes: cache, tlb
   - Repeats: 3 runs per configuration
   - **Total runs**: 3 contexts × 3 generations × 2 probes × 3 repeats = 54 runs

2. **Decoding Strategy Attack** (BTB + PHT probes)
   - Decoding modes: greedy, sampling
   - Context: 512 tokens (fixed)
   - Generation: 128 tokens
   - Probes: btb, pht
   - Repeats: 3 runs per configuration
   - **Total runs**: 2 decoding modes × 2 probes × 3 repeats = 12 runs

3. **Temperature Sweep** (for sampling analysis)
   - Temperatures: 0.1, 0.5, 1.0, 1.5
   - Context: 512 tokens
   - Generation: 128 tokens
   - Probe: pht
   - Repeats: 3 runs per configuration
   - **Total runs**: 4 temperatures × 3 repeats = 12 runs

4. **Prompt Semantics Attack** (PHT probe)
   - Prompt types: Math, Code, Natural Language, Custom
   - Context: 512 tokens
   - Generation: 256 tokens (longer for semantic leakage)
   - Probe: pht
   - Repeats: 3 runs per configuration
   - **Total runs**: 4 prompt types × 3 repeats = 12 runs

**Total Sweep**: ~90 runs

**Expected Runtime:**
- Single run: ~15-45 seconds (DeepSeek is slower than TinyLLaMA due to 7.3× more parameters)
- Full sweep: ~1.5-3 hours depending on hardware

**Note**: The script continues even if individual runs fail, logging errors for later inspection.

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
- `extreme_contexts_comparison_deepseek-8b.png`: Side-by-side comparison of 128 vs 2048 token contexts

### Expected Results (DeepSeek-8B)

Based on experimental evaluation:

**Context Size Classification:**
- **Accuracy**: ~87.2% (3-class: 128/512/2048 tokens)
- **Improvement over TinyLLaMA**: +6.7 percentage points (80.5% → 87.2%)
- **Best Probe**: Cache (LLC contention strongly correlates with larger KV-cache)
- **F1 Scores**: 128-token: 0.89, 512-token: 0.84, 2048-token: 0.88
- **Key Features**: Kurtosis (1498 for 2048-token), p99, standard deviation
- **Amplification**: Larger model → larger KV-cache → stronger cache contention signals

**Decoding Strategy Classification:**
- **Accuracy**: ~96.9% (2-class: greedy/sampling)
- **Improvement over TinyLLaMA**: +6.9 percentage points (90.0% → 96.9%)
- **Best Probe**: BTB (branch prediction patterns more pronounced in larger model)
- **F1 Scores**: Greedy: 0.95, Sampling: 0.98 (nearly perfect, balanced performance)
- **Key Features**: Kurtosis, skewness, median
- **Observation**: Balanced classification (no asymmetry unlike TinyLLaMA)

**Prompt Semantics Classification:**
- **Accuracy**: ~64.6% (4-class: Math/Code/NL/Custom)
- **Improvement over TinyLLaMA**: +3.2 percentage points (61.4% → 64.6%)
- **Best Probe**: PHT (control flow patterns vary by content type)
- **F1 Scores**: Math: 0.63, Code: 0.58, NL: 0.71, Custom: 0.93
- **Per-Class Improvements**:
  - Code: 2.7× better (0.22 → 0.58)
  - Math: 1.5× better (0.42 → 0.63)
  - NL: 1.5× better (0.47 → 0.71)
- **Observation**: Custom prompts still most distinguishable; larger model shows better semantic separation

**Distribution Characteristics (Amplification Effects):**
- **Skewness**: 38.7 (vs 21.7 for TinyLLaMA) - **1.8× amplification**
- **Kurtosis**: 1498 (vs 666 for TinyLLaMA) - **2.2× amplification**
- **Cycle Range**: 300-80,000+ cycles (wider range than TinyLLaMA)
- **Interpretation**: Larger models produce more extreme timing outliers, creating stronger distinguishability

**Key Insight:**
DeepSeek's 7.3× larger parameter count (8B vs 1.1B) amplifies microarchitectural side-channel leakage across all attack types. The increased computational complexity and memory footprint create more pronounced contention patterns, paradoxically making larger "more capable" models **easier to attack**.

## Understanding the Attack

### Why Amplification Occurs

**Architectural Scaling Effects:**
- **Larger KV-Cache**: 8B model stores 7.3× more key-value pairs per token
  - Context size leakage: More cache lines accessed → stronger contention
  - Result: Better cache probe classification accuracy
  
- **More Computation**: 32 layers (vs 22 in TinyLLaMA), 4096 hidden dim (vs 2048)
  - Longer execution time → more probe samples → better statistics
  - Result: Reduced measurement noise

- **Complex Branching**: Larger vocabulary, more attention heads
  - Sampling generates more diverse branch patterns
  - Result: BTB/PHT probes see clearer differences

**Distribution Amplification:**
DeepSeek shows **extreme timing outliers**:
- Kurtosis 1498 indicates heavy-tailed distributions with rare but massive spikes
- These spikes occur during specific inference phases (attention, FFN, sampling)
- Attack exploits: Even 1-2 extreme outliers per run enable accurate classification

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

**Key Observation**: Kurtosis is the **most discriminative feature** for DeepSeek:
- Context: Kurtosis ranges from 200 (128-token) to 1498 (2048-token)
- Decoding: Greedy has lower kurtosis than sampling
- This single feature often achieves 70-80% accuracy alone

## Important Notes

**System Considerations:**
- **SMT Required**: Attack only works on SMT siblings sharing physical core
- **CPU Pinning**: `taskset` ensures victim/attacker on correct cores
- **Noise**: Real systems have background noise; use multiple repeats
- **Warmup**: First ~500 probe iterations discarded (warmup effects)
- **Frequency Scaling**: CPU frequency sampled to detect DVFS interference
- **Memory**: DeepSeek requires ~6-7GB RAM for model + inference buffers

**Performance vs Security Trade-off:**
- Larger models are **more capable** but **more vulnerable** to side channels
- Production deployments (GPT-4 scale, 100B+ parameters) likely face even greater risk
- Mitigation costs scale with model size (more padding, isolation overhead)

**Security Implications:**
- Unprivileged attacker (no special permissions required)
- Co-location sufficient (cloud multi-tenancy scenarios)
- Software-only mitigation difficult (architectural issue)
- Defenses: Disable SMT, context padding, decoding normalization

**Experimental Limitations:**
- Controlled setup (isolated system, minimal background noise)
- Fixed generation lengths (variable length adds complexity)
- Single model family (LLaMA architecture)
- Limited prompt diversity (4 semantic categories)
- CPU-only inference (GPU/TPU may show different patterns)

## Comparison: DeepSeek-8B vs TinyLLaMA-1.1B

| Metric | TinyLLaMA-1.1B | DeepSeek-8B | Amplification |
|--------|----------------|-------------|---------------|
| **Parameters** | 1.1B | 8B | 7.3× |
| **Model Size (Q4)** | ~600MB | ~5GB | 8.3× |
| **Context Accuracy** | 80.5% | 87.2% | +6.7pp |
| **Decoding Accuracy** | 90.0% | 96.9% | +6.9pp |
| **Semantics Accuracy** | 61.4% | 64.6% | +3.2pp |
| **Skewness** | 21.7 | 38.7 | 1.8× |
| **Kurtosis** | 666 | 1498 | 2.2× |
| **Inference Time** | ~10-20s | ~20-40s | 2-3× |

**Takeaway**: Larger models amplify side-channel leakage, making them **easier targets** for inference parameter extraction attacks.

## Troubleshooting

**"llama-cli not found":**
- Verify build: `ls ~/llama.cpp/build/bin/llama-cli`
- Rebuild: `cd ~/llama.cpp/build && cmake --build . --config Release`
- Update `VICTIM_BIN` path in scripts

**"Model file not found":**
- Check path: `ls ~/llama.cpp/models/deepseek-llama8b-q4.gguf`
- Verify download completed (file should be ~5GB)
- Update `MODEL` path in `run_sweeps.sh`

**"Cannot detect SMT siblings":**
- Verify SMT enabled in BIOS
- Check: `lscpu | grep "Thread(s) per core"`
- Should show "2" for SMT enabled
- Manual check: `cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list`

**"Permission denied" on CPU pinning:**
- Run with sufficient privileges or adjust `taskset` permissions
- Some systems restrict CPU affinity for non-root users
- Try: `sudo -E bash run_sweeps.sh` (preserves environment)

**Low classification accuracy:**
- Ensure sufficient repeats (3+ per config minimum)
- Check for system noise: `top`, `htop` (close heavy processes)
- Disable turbo boost for consistent frequencies
- Verify victim/attacker on true SMT siblings
- Increase warmup discard: `--warmup-discard 1000`

**Probe crashes or hangs:**
- Reduce `--iters` if memory constrained (try 1000)
- Check probe compilation: `make clean && make -C attacker`
- Verify CPU IDs are valid: `cat /proc/cpuinfo | grep processor`

**Out of memory errors:**
- DeepSeek requires ~6-7GB RAM
- Close other applications
- Consider using smaller context (128 or 512 tokens)
- Monitor with: `watch -n 1 free -h`

**Slow inference:**
- Enable BLAS: Rebuild llama.cpp with `-DGGML_BLAS=ON`
- Consider smaller quantization: Q4_0 instead of Q4_K_M
- Reduce generation length: `--npredict 64` instead of 256

## Quick Reference

**Setup:**
```bash
# Build probes
make -C attacker

# Find SMT siblings
python3 -c "from driver.run_utils import detect_siblings; print(detect_siblings())"

# Set environment
export VICTIM_BIN=~/llama.cpp/build/bin/llama-cli
export MODEL_PATH=~/llama.cpp/models/deepseek-llama8b-q4.gguf
export VCPU=0 ACPU=56
```

**Run Experiments:**
```bash
# Single run (context size)
python3 driver/driver.py --victim-bin "$VICTIM_BIN" --model "$MODEL_PATH" \
  --ctx 512 --npredict 64 --decoding greedy --prompt "Test prompt" \
  --victim-cpu $VCPU --attacker-cpu $ACPU \
  --probe-bin ./attacker/cache_probe --probe cache --iters 2000 --seed 42 --repeat 1

# Full sweep (all experiments)
bash run_sweeps.sh
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

# Check for failed runs
grep -i error logs/runs/*/meta.json
```

**Performance Monitoring:**
```bash
# Monitor during runs
watch -n 1 'ps aux | grep llama-cli; free -h'

# Check CPU frequency stability
watch -n 0.5 'grep MHz /proc/cpuinfo | head -2'
```

## Citation

If you use this code or methodology, please cite:

```bibtex
@inproceedings{jani2025sidechannel,
  title={Microarchitectural Side-Channel Attacks on CPU-Based LLM Inference},
  author={Jani, Devesh and Joshi, Misha},
  booktitle={ECE 592 Final Project},
  year={2025},
  note={DeepSeek-R1-Distill-Llama-8B experiments}
}
```

## Related Work

See also:
- **TinyLlama/** - Baseline experiments with TinyLLaMA-1.1B model
- **paper/** - Full research paper with detailed methodology and results
- Original llama.cpp: https://github.com/ggerganov/llama.cpp
- DeepSeek model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
