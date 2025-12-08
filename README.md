# Microarchitectural Side-Channel Attacks on CPU-Based LLM Inference

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](ECE592-Final_Project_Report.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Authors:** Devesh Jani, Misha Joshi  
**Course:** ECE 592 - Advanced Topics in Computer Engineering  
**Institution:** North Carolina State University  
**Date:** December 2025

## Overview

This project demonstrates that **unprivileged co-located attackers can infer LLM inference parameters** through microarchitectural side-channel attacks on SMT (Simultaneous Multi-Threading) sibling cores. We target CPU-based inference using `llama.cpp` and show that larger models paradoxically exhibit **amplified leakage**, making them easier to attack.

### Key Findings

| Attack Target | TinyLLaMA-1.1B | DeepSeek-8B | Improvement |
|--------------|----------------|-------------|-------------|
| **Context Size** (3-class) | 80.5% | 87.2% | +6.7pp |
| **Decoding Strategy** (2-class) | 90.0% | 96.9% | +6.9pp |
| **Prompt Semantics** (4-class) | 61.4% | 64.6% | +3.2pp |
| **Distribution Kurtosis** | 666 | 1498 | **2.2× amplification** |
| **Distribution Skewness** | 21.7 | 38.7 | **1.8× amplification** |

**Main Insight:** The 7.3× larger DeepSeek model shows 1.8-2.2× stronger side-channel signals, demonstrating that model scaling amplifies microarchitectural leakage.

## Attack Methodology

### Threat Model
- **Attacker**: Unprivileged process on SMT sibling core
- **Victim**: LLM inference process (llama.cpp)
- **Attack Vector**: Cache, TLB, BTB, and PHT contention timing
- **Goal**: Infer context size, decoding strategy, and prompt semantics

### Probes Implemented
1. **Cache Probe**: Prime+Probe on Last Level Cache
2. **TLB Probe**: Translation Lookaside Buffer contention
3. **BTB Probe**: Branch Target Buffer collision detection
4. **PHT Probe**: Pattern History Table interference

### Classification Pipeline
- **Features**: Mean, median, std, p10/p90/p99, skewness, kurtosis (8 per run)
- **Classifier**: Random Forest (100 trees)
- **Training**: 80/20 split with stratification
- **Evaluation**: Confusion matrices, F1 scores, feature importance

## Repository Structure

```
ECE592-FinalProject/
├── TinyLlama/              # TinyLLaMA-1.1B experiments (baseline)
│   ├── driver/             # Orchestration scripts
│   ├── attacker/           # Microarchitectural probes (C)
│   ├── analysis/           # Statistical analysis & ML classifiers
│   ├── logs/               # Experimental data (runs & index)
│   └── README.md           # Detailed setup & usage
│
├── DeepSeek/               # DeepSeek-8B experiments (scaling study)
│   ├── driver/             # Orchestration scripts
│   ├── attacker/           # Microarchitectural probes (C)
│   ├── analysis/           # Statistical analysis & ML classifiers
│   ├── logs/               # Experimental data (runs & index)
│   ├── run_sweeps.sh       # Automated experiment suite
│   └── README.md           # Detailed setup & usage
│
├── paper/                  # LaTeX source & figures
│   ├── main.tex            # IEEE conference paper (12 references)
│   ├── references.bib      # Bibliography
│   ├── figs/               # All figures (44 PNG files)
│   └── *.zip               # Overleaf-ready packages
│
└── ECE592-Final_Project_Report.pdf  # Complete research paper
```

## Quick Start

### Prerequisites
- Linux with SMT-enabled CPU
- Python 3.9+ (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`)
- `gcc`, `make` for building probes
- `llama.cpp` compiled with target model (TinyLLaMA or DeepSeek GGUF)

### 1. Build Attack Probes
```bash
# For TinyLLaMA experiments
cd TinyLlama
make -C attacker

# For DeepSeek experiments
cd DeepSeek
make -C attacker
```

### 2. Find SMT Siblings
```bash
python3 -c "from driver.run_utils import detect_siblings; print(detect_siblings())"
# Example output: {0: [0, 56], 1: [1, 57], ...}
# Use pairs like victim_cpu=0, attacker_cpu=56
```

### 3. Run Experiments

**TinyLLaMA (baseline):**
```bash
cd TinyLlama

# Set paths
export VICTIM_BIN=/path/to/llama.cpp/main
export MODEL_PATH=/path/to/tinyllama-1.1b-q4_0.gguf

# Run context size sweep
bash scripts/run_ctx_decoding_sweep.sh

# Run semantics sweep
bash scripts/run_btb_prompt_sweep.sh
```

**DeepSeek (scaling study):**
```bash
cd DeepSeek

# Configure run_sweeps.sh with your paths
export VICTIM_BIN=/path/to/llama.cpp/build/bin/llama-cli
export MODEL_PATH=/path/to/deepseek-llama8b-q4.gguf

# Run full experiment suite (~90 runs, 1.5-3 hours)
bash run_sweeps.sh
```

### 4. Analyze Results
```bash
python3 analysis/analysis.py \
  --logs-dir logs \
  --output-dir analysis/figs \
  --warmup-discard 500
```

**Outputs:**
- `run_statistics.csv`: Per-run aggregate statistics
- `confusion_matrix_*.png`: Classification performance
- `feature_importance_*.png`: Most discriminative features
- `hist_cycles_*.png`, `boxplot_*.png`: Distribution visualizations
- `extreme_contexts_comparison_*.png`: Side-by-side 128 vs 2048 token comparison

## Key Results

### Context Size Classification
- **Best Probe:** Cache (LLC contention scales with KV-cache size)
- **DeepSeek Advantage:** Larger KV-cache → stronger cache signals
- **Key Feature:** Kurtosis (extreme outliers at 2048 tokens)

### Decoding Strategy Classification
- **Best Probe:** BTB (deterministic vs stochastic branching)
- **DeepSeek Advantage:** More complex sampling → clearer branch patterns
- **Observation:** Nearly perfect classification (96.9% for DeepSeek)

### Prompt Semantics Classification
- **Best Probe:** PHT (control flow varies by content type)
- **Challenge:** Generic categories (Math/Code/NL) show weak signals
- **Strong Signal:** Custom prompts highly distinguishable (93% recall)

### Amplification Analysis
DeepSeek-8B shows **extreme timing outliers**:
- Kurtosis up to **1498** (vs 666 for TinyLLaMA)
- Skewness up to **38.7** (vs 21.7 for TinyLLaMA)
- Cycle range: 300-80,000+ (vs 200-50,000 for TinyLLaMA)

**Implication:** Production-scale models (GPT-4 class, 100B+ parameters) likely face even greater vulnerability.

## Security Implications

### Real-World Attack Scenarios
1. **Service Fingerprinting**: Identify which LLM service/configuration is deployed
2. **Competitive Intelligence**: Expose quality-cost trade-offs in multi-tenant environments
3. **Privacy Violations**: Coarse-grained semantic classification without plaintext access
4. **Stepping Stone**: Enable model extraction and adversarial attacks

### Recommended Mitigations
- **SMT Isolation**: Disable SMT or dedicate physical cores (50% throughput cost)
- **Context Padding**: Always use max context (2-8× latency/memory overhead)
- **Decoding Normalization**: Fixed decoding mode (reduces model expressivity)
- **Hardware Partitioning**: Intel CAT/MBA for cache isolation (limited availability)

## Paper

The complete research paper is available in [`ECE592-Final_Project_Report.pdf`](ECE592-Final_Project_Report.pdf) and includes:
- Comprehensive threat model and attack methodology
- Detailed experimental setup and parameter sweeps
- Statistical analysis of timing distributions
- Cross-model comparison and amplification study
- Discussion of mitigation strategies and limitations

**LaTeX Source:** Available in `paper/` directory with:
- 12 carefully selected references
- 44 publication-quality figures
- Detailed appendix with team contributions
- Overleaf-ready ZIP packages

## Reproducibility

### Hardware Used
- Intel Xeon processors with SMT enabled
- Ubuntu 22.04 LTS
- 16+ GB RAM (6-7GB needed for DeepSeek)

### Software Versions
- Python 3.9+
- llama.cpp (latest stable, commit hash in experiment logs)
- scikit-learn 1.3+
- Models: TinyLLaMA-1.1B-Chat-v1.0 (Q4_0), DeepSeek-R1-Distill-Llama-8B (Q4_K_M)

### Data Availability
Full experimental logs are preserved in `TinyLlama/logs/` and `DeepSeek/logs/`:
- `index.csv`: Master index of all runs (700+ experiments)
- `runs/<run_id>/`: Per-run probe data, metadata, timings, frequency samples

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{jani2025sidechannel,
  title={Microarchitectural Side-Channel Attacks on CPU-Based LLM Inference},
  author={Jani, Devesh and Joshi, Misha},
  booktitle={ECE 592 Advanced Topics in Computer Engineering - Final Project},
  institution={North Carolina State University},
  year={2025}
}
```

## Team Contributions

**Devesh Jani:**
- TinyLLaMA context size and decoding strategy attacks
- DeepSeek context size attack and amplification analysis
- Cache and TLB probe implementation and data collection
- Statistical analysis and visualization pipeline

**Misha Joshi:**
- TinyLLaMA prompt semantics attack
- DeepSeek decoding strategy and semantics attacks
- BTB and PHT probe implementation and data collection
- Cross-model comparison and semantic classification

**Shared:**
- Experimental infrastructure (driver, orchestration)
- Random Forest classifier training and evaluation
- Paper writing and result interpretation
- Mitigation analysis and security implications

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- llama.cpp community for CPU-optimized inference
- TinyLLaMA and DeepSeek teams for open-source models
- ECE 592 course staff for project guidance

## Related Work

- **Cache Side Channels**: Osvik et al. (Prime+Probe), Yarom & Falkner (Flush+Reload)
- **ML Side Channels**: Yan et al. (cache attacks on NNs), Chen et al. (RNN input inference)
- **LLM Security**: Carlini et al. (training data extraction), Shokri et al. (membership inference)
- **Speculative Execution**: Kocher et al. (Spectre), Gras et al. (TLB attacks)

## Contact

For questions or collaborations:
- Devesh Jani: dhjani2@ncsu.edu
- Misha Joshi: mjoshi7@ncsu.edu

---

**Disclaimer:** This research is for educational and academic purposes. Unauthorized use of side-channel attacks against production systems may violate terms of service and applicable laws.
