"""
Minimal orchestration script to run a victim (llama.cpp main) and an attacker probe
on SMT siblings, capture logs in the standardized structure, and write metadata.

This is a skeleton: adapt commands/paths to your environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import signal
import time
from pathlib import Path

from run_utils import (
    FreqSampler,
    append_index_row,
    make_run_dir,
    make_run_id,
    model_tag_from_file,
    run_with_taskset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path.cwd()), help="project root")
    ap.add_argument("--victim-bin", type=str, required=True, help="path to llama.cpp main")
    ap.add_argument("--model", type=str, required=True, help="model gguf path")
    ap.add_argument("--quant", type=str, default="q4_0")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--npredict", type=int, default=64)
    ap.add_argument("--decoding", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="repeat index (1-based)")
    ap.add_argument("--victim-cpu", type=int, required=True)
    ap.add_argument("--attacker-cpu", type=int, required=True)
    ap.add_argument("--probe-bin", type=str, required=True, help="path to probe binary")
    ap.add_argument("--probe", type=str, choices=["cache","tlb","btb","pht","rob"], required=True)
    ap.add_argument("--iters", type=int, default=2000, help="probe iterations (if applicable)")
    ap.add_argument("--warmup-ms", type=int, default=100)
    ap.add_argument("--freq-sample-cpu", type=int, default=None, help="CPU id for freq sampling (default: attacker cpu)")
    ap.add_argument("--prompt-label", type=str, default="custom")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    now = dt.datetime.utcnow()
    run_id = make_run_id(
        now_utc=now,
        victim_cpu=args.victim_cpu,
        attacker_cpu=args.attacker_cpu,
        probe=args.probe,
        model_path=args.model,
        quant=args.quant,
        ctx=args.ctx,
        npredict=args.npredict,
        decoding=args.decoding,
        seed=args.seed,
        repeat_idx=args.repeat,
    )
    run_dir = make_run_dir(root, run_id)

    # Frequency sampler
    freq_cpu = args.freq_sample_cpu if args.freq_sample_cpu is not None else args.attacker_cpu
    fs = FreqSampler(cpu=freq_cpu, out_csv=run_dir / "freq.csv", interval_sec=0.02)
    fs.start()

    timings = {}
    t0 = time.time()

    # Build victim command
    # victim_cmd = [
    #     args.victim_bin,
    #     "-m", args.model,
    #     "-t", "1",
    #     "-c", str(args.ctx),
    #     "-n-predict", str(args.npredict),
    #     "-p", args.prompt,
    #     "--seed", str(args.seed),
    # ]

    victim_cmd = [
        args.victim_bin,
        "-m", args.model,
        "-t", "1",
        "-c", str(args.ctx),
        "-n", str(args.npredict),
        "-p", args.prompt,
        "--seed", str(args.seed),
    ]
    if args.decoding == "greedy":
        victim_cmd += ["--temp", "0"]
    else:
        victim_cmd += ["--temp", str(args.temp if args.temp is not None else 1.0)]
        if args.top_k is not None:
            victim_cmd += ["--top-k", str(args.top_k)]
        if args.top_p is not None:
            victim_cmd += ["--top-p", str(args.top_p)]

    victim_out = run_dir / "victim_stdout.txt"
    victim = run_with_taskset(victim_cmd, args.victim_cpu, victim_out)

    try:
        time.sleep(args.warmup_ms / 1000.0)
        timings["victim_warmup_ms"] = args.warmup_ms

        # Build attacker command (append cpu or iters if the probe expects)
        # Convention: our probes accept the pinned CPU id as first arg; adjust as needed.
        probe_cmd = [args.probe_bin, str(args.attacker_cpu), str(args.iters)]
        attacker_out = run_dir / "attacker_stdout.txt"
        attacker = run_with_taskset(probe_cmd, args.attacker_cpu, attacker_out)

        attacker.wait()
        timings["probe_ms"] = int((time.time() - (t0 + args.warmup_ms / 1000.0)) * 1000)

    finally:
        # Stop victim if still running
        try:
            if victim.poll() is None:
                victim.send_signal(signal.SIGINT)
                try:
                    victim.wait(timeout=5)
                except Exception:
                    victim.kill()
        except Exception:
            pass
        # Stop freq sampler
        fs.stop()
        try:
            fs.join(timeout=1)
        except (TypeError, AttributeError):
            # Python 3.9 threading bug workaround
            pass

    timings["total_ms"] = int((time.time() - t0) * 1000)

    # Write meta.json
    meta = {
        "run_id": run_id,
        "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": os.uname().nodename.split(".")[0],
        "victim_cpu": args.victim_cpu,
        "attacker_cpu": args.attacker_cpu,
        "probe": args.probe,
        "model": args.model,
        "model_tag": model_tag_from_file(args.model),
        "quant": args.quant,
        "ctx": args.ctx,
        "npredict": args.npredict,
        "decoding": args.decoding,
        "temp": args.temp if args.decoding == "sample" else 0.0,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "repeat_idx": args.repeat,
        "prompt": args.prompt,
        "prompt_label": args.prompt_label,
        "prompt_hash": "",  # fill if you hash inputs
        "durations_ms": timings,
        "versions": {"driver": "v0.1", "probe_git": None, "analysis_git": None, "llama_cpp_git": None},
    }
    write_json(run_dir / "meta.json", meta)
    write_json(run_dir / "timings.json", timings)

    # Append index.csv row
    append_index_row(
        root / "logs" / "index.csv",
        {
            "run_id": run_id,
            "timestamp_utc": meta["timestamp_utc"],
            "path": str(run_dir),
            "host": meta["host"],
            "victim_cpu": args.victim_cpu,
            "attacker_cpu": args.attacker_cpu,
            "probe": args.probe,
            "model_tag": meta["model_tag"],
            "quant": args.quant,
            "ctx": args.ctx,
            "npredict": args.npredict,
            "decoding": args.decoding,
            "temp": meta["temp"],
            "seed": args.seed,
            "repeat_idx": args.repeat,
            "prompt_label": meta["prompt_label"],
            "prompt_hash": meta["prompt_hash"],
        },
    )

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
