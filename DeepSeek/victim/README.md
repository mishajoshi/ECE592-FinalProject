# Victim Setup - TinyLLaMA Inference

This directory contains instructions for setting up and running TinyLLaMA (via llama.cpp) as the victim workload for SMT contention experiments.

## Quick Start

### 1. Install llama.cpp

Clone and build llama.cpp:

```bash
cd ~/
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

For better performance, build with optimizations:

```bash
# Use OpenBLAS
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

# OR use CUDA (if NVIDIA GPU available)
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

### 2. Download TinyLLaMA Model

Download a quantized TinyLLaMA model:

```bash
cd ~/llama.cpp
mkdir -p models

# Download TinyLLaMA 1.1B Q4_0 quantized model (recommended for experiments)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -O models/tinyllama-1.1b-q4_0.gguf

# Optional: Download other quantization levels for sweep experiments
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_0.gguf \
  -O models/tinyllama-1.1b-q5_0.gguf

wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -O models/tinyllama-1.1b-q8_0.gguf
```

### 3. Test the Model

Run a simple inference test:

```bash
cd ~/llama.cpp
./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf \
  -t 1 \
  -c 512 \
  -n 64 \
  --temp 0 \
  -p "Explain caching with an example."
```

**Important parameters:**
- `-t 1`: Single thread (required for clean SMT experiments)
- `-c 512`: Context size (working set parameter)
- `-n 64`: Number of tokens to predict
- `--temp 0`: Greedy decoding (deterministic)
- `-p "..."`: Input prompt

## Running with CPU Pinning

For experiments, pin the victim to a specific logical CPU:

```bash
# Pin to CPU 2 (check SMT siblings first)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf \
  -t 1 -c 512 -n 64 --temp 0 \
  -p "Explain caching with an example."
```

## Checking SMT Siblings

Find SMT siblings on your system:

```bash
# List thread siblings for all CPUs
grep . /sys/devices/system/cpu/cpu*/topology/thread_siblings_list

# Example output:
# /sys/devices/system/cpu/cpu0/topology/thread_siblings_list:0,4
# /sys/devices/system/cpu/cpu1/topology/thread_siblings_list:1,5
# /sys/devices/system/cpu/cpu2/topology/thread_siblings_list:2,6
# /sys/devices/system/cpu/cpu3/topology/thread_siblings_list:3,7

# In this example, CPUs (0,4), (1,5), (2,6), (3,7) are SMT sibling pairs
```

**For experiments:** If you run victim on CPU 2, run attacker probe on CPU 6 (its SMT sibling).

## Parameter Sweep Examples

### S1: Working Set / Context Sweep

Test different memory working sets:

```bash
# Small context (128 tokens)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 128 -n 16 --temp 0 -p "Test."

# Medium context (512 tokens)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 512 -n 64 --temp 0 -p "Test."

# Large context (2048 tokens)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 2048 -n 256 --temp 0 -p "Test."
```

### S2: Quantization Width Sweep

Test different quantization levels:

```bash
# Q4_0 quantization (4-bit)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 512 -n 64 --temp 0 -p "Test."

# Q5_0 quantization (5-bit)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q5_0.gguf -t 1 -c 512 -n 64 --temp 0 -p "Test."

# Q8_0 quantization (8-bit)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q8_0.gguf -t 1 -c 512 -n 64 --temp 0 -p "Test."
```

### S3: Access Pattern / Decoding Sweep

Test different decoding strategies:

```bash
# Greedy decoding (deterministic, predictable branches)
 taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 512 -n 64 --temp 0 -p "Test."

# Sampling decoding (stochastic, unpredictable branches)
taskset -c 2 ./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -t 1 -c 512 -n 64 \
  --temp 1.0 --top-k 40 --top-p 0.95 -p "Test."
```

## Prompts for Experiments

Use consistent prompts for reproducibility:

**Low diversity (short, simple):**
```
"Test."
"Hello world."
"Explain caching."
```

**High diversity (longer, complex):**
```
"Explain the concept of cache coherence in modern multicore processors with detailed examples."
"Describe the differences between direct-mapped, set-associative, and fully-associative caches."
```

## Using with the Driver

The driver script (`driver/driver.py`) automates victim execution. Example:

```bash
cd /mnt/ncsudrive/d/dhjani2/microarch_finalproject

python3 driver/driver.py \
  --root . \
  --victim-bin ~/llama.cpp/build/bin/llama-cli \
  --model ~/llama.cpp/models/tinyllama-1.1b-q4_0.gguf \
  --quant q4_0 \
  --ctx 512 \
  --npredict 64 \
  --decoding greedy \
  --victim-cpu 2 \
  --attacker-cpu 6 \
  --probe-bin ./attacker/cache_probe \
  --probe cache \
  --repeat 1
```

## Performance Tuning

For best experimental results:

1. **Set CPU governor to performance:**
   ```bash
   sudo cpupower frequency-set -g performance
   ```

2. **Disable turbo boost (optional, for consistency):**
   ```bash
   echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

3. **Isolate CPUs (advanced):**
   ```bash
   # Add to kernel boot parameters: isolcpus=2,6
   # Then reboot
   ```

4. **Close background processes:**
   ```bash
   # Minimize system load during experiments
   killall chrome firefox slack discord
   ```

## Troubleshooting

**Issue:** `llama-cli: command not found`
```
Solution: Binary is in build/bin/ directory. Use full path: ~/llama.cpp/build/bin/llama-cli
```

**Issue:** Model not loading
```
Solution: Check model path and ensure .gguf file is complete
```

**Issue:** Slow inference
```
Solution: Use smaller model or rebuild with OpenBLAS/CUDA (see cmake options above)
```

**Issue:** Can't pin to CPU
```
Solution: Check if CPU exists: cat /proc/cpuinfo | grep processor
```

**Issue:** No SMT siblings
```
Solution: Enable Hyper-Threading in BIOS
```

## References

- llama.cpp: https://github.com/ggerganov/llama.cpp
- TinyLLaMA models: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- Model quantization guide: https://github.com/ggerganov/llama.cpp#quantization
