# DeepSeek SMT Leakage Attack

Experiments to measure side-channel leakage when running DeepSeek R1 Distill Llama 8B on SMT siblings. The victim runs `llama.cpp` on one logical CPU while attacker probes run on the sibling to capture cache/branch/TLB effects. `run_sweeps.sh` orchestrates all sweeps and writes logs in `logs/`.

## Prerequisites
- Linux with SMT (hyper-threading) enabled and `taskset` available.
- Build tools: `gcc`, `make`, `cmake`, `git`.
- Python 3.9+ with `pip` (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`).
- Disk space for the DeepSeek model (≈5–6 GB for Q4 GGUF).

## 1) Build attacker probes
```bash
cd /mnt/ncsudrive/m/mjoshi7/FinalProject/microarch_finalproject
make -C attacker
```
This produces `attacker/cache_probe`, `tlb_probe`, `btb_probe`, and `pht_probe`.

## 2) Install llama.cpp
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j"$(nproc)"
```
If you want BLAS or CUDA, add `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` or `-DGGML_CUDA=ON` to the first `cmake` command.

## 3) Download DeepSeek 8B GGUF
```bash
cd ~/llama.cpp
mkdir -p models
wget https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/deepseek-r1-distill-llama-8b.Q4_K_M.gguf \
  -O models/deepseek-llama8b-q4.gguf
```
This path matches the default `MODEL` setting in `run_sweeps.sh`. Adjust the filename there if you pick a different quant.

## 4) Pinning and CPU pairing
Find SMT siblings (example output: `0,56` means CPU 0 and 56 are siblings):
```bash
grep . /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
```
Set `VICTIM_CPU` and `ATTACKER_CPU` in `run_sweeps.sh` to a sibling pair (victim on the first, attacker on the second).

## 5) Configure and run sweeps
Open `run_sweeps.sh` and update these variables near the top if needed:
- `PROJECT_ROOT` (repo path)
- `VICTIM_BIN` (path to `llama.cpp` `llama-cli`)
- `MODEL` (path to the DeepSeek GGUF)
- `VICTIM_CPU` / `ATTACKER_CPU`

Run the full DeepSeek experiment suite:
```bash
bash run_sweeps.sh
```
The script performs:
- Context-size sweep (CTX 128/512/2048, NPREDICT 16/64/256)
- Decoding sweep (greedy vs sampling) across BTB and PHT probes
- Temperature sweep for sampling
- Prompt semantics sweep with long prompts and PHT probe

If any run fails, the script continues to the next sweep. Logs are written even when a run reports an error.

## 6) Outputs and analysis
- Raw logs and metadata: `logs/runs/<run_id>/` (probe stdout, victim stdout, meta.json, timings.json, freq.csv).
- Run index: `logs/index.csv`.
- Quick analysis and plots:
  ```bash
  cd /mnt/ncsudrive/m/mjoshi7/FinalProject/microarch_finalproject
  python3 analysis/analysis.py --logs-dir logs --output-dir analysis/figs
  ```
  Generated figures land in `analysis/figs/`.

## 7) Troubleshooting
- Ensure `llama-cli` exists at `VICTIM_BIN`; rerun the `cmake` build if not.
- Verify the model path in `MODEL`; the file must match the chosen quantization.
- Rebuild probes with `make -C attacker clean all` if binaries are missing or stale.
- Use `taskset -c <cpu>` manually on a small test prompt to confirm `llama.cpp` runs correctly before launching sweeps.
