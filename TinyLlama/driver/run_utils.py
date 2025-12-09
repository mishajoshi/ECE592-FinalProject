"""
Utilities for run ID generation, logging structure, metadata, perf wrapping,
and frequency sampling for the SMT contention experiment.

Safe to use before real probes exist; integrates with a generic driver.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence


def _slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s/:+.,()-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def model_tag_from_file(model_path: str) -> str:
    base = os.path.basename(model_path)
    base = re.sub(r"\.gguf$", "", base)
    base = re.sub(r"-q[0-9]_[0-9]$", "", base)
    base = base.replace(".", "p")
    return _slug(base)


def make_run_id(
    now_utc: Optional[dt.datetime],
    victim_cpu: int,
    attacker_cpu: int,
    probe: str,
    model_path: str,
    quant: str,
    ctx: int,
    npredict: int,
    decoding: str,  # "greedy" or "sample"
    seed: int,
    repeat_idx: int,
    host: Optional[str] = None,
) -> str:
    ts = (now_utc or dt.datetime.utcnow()).strftime("%Y%m%dT%H%M%SZ")
    host = _slug((host or socket.gethostname()).split(".")[0])
    mdl = model_tag_from_file(model_path)
    dec = "g" if decoding == "greedy" else "s"
    fields = [
        ts,
        host,
        f"v{victim_cpu}a{attacker_cpu}",
        _slug(probe),
        mdl,
        _slug(quant),
        f"c{ctx}",
        f"n{npredict}",
        dec,
        f"seed{seed}",
        f"r{repeat_idx:03d}",
    ]
    return "-".join(fields)


def make_run_dir(root: Path, run_id: str) -> Path:
    run_dir = root / "logs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_index_row(index_csv: Path, row: Dict[str, object]) -> None:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not index_csv.exists()
    cols = [
        "run_id",
        "timestamp_utc",
        "path",
        "host",
        "victim_cpu",
        "attacker_cpu",
        "probe",
        "model_tag",
        "quant",
        "ctx",
        "npredict",
        "decoding",
        "temp",
        "seed",
        "repeat_idx",
        "prompt_label",
        "prompt_hash",
    ]
    with index_csv.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write(",".join(cols) + "\n")
        vals = [str(row.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")


class FreqSampler(threading.Thread):
    """Sample a CPU's scaling_cur_freq into a CSV file at fixed intervals."""

    def __init__(self, cpu: int, out_csv: Path, interval_sec: float = 0.02):
        super().__init__(daemon=True)
        self.cpu = cpu
        self.out_csv = out_csv
        self.interval = interval_sec
        self._stop = threading.Event()

    def run(self) -> None:
        path = Path(f"/sys/devices/system/cpu/cpu{self.cpu}/cpufreq/scaling_cur_freq")
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", encoding="utf-8") as f:
            f.write("ts_ns,freq_khz\n")
            while not self._stop.is_set():
                ts_ns = time.time_ns()
                try:
                    with path.open("r", encoding="utf-8") as pf:
                        freq = pf.read().strip()
                    f.write(f"{ts_ns},{freq}\n")
                except FileNotFoundError:
                    # cpufreq not available; write -1 once and stop
                    f.write(f"{ts_ns},-1\n")
                    break
                time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


def run_with_taskset(cmd: Sequence[str], cpu: int, stdout_path: Path) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    fout = stdout_path.open("w", buffering=1, encoding="utf-8")
    proc = subprocess.Popen([
        "taskset", "-c", str(cpu), *cmd
    ], stdout=fout, stderr=subprocess.STDOUT, text=True)
    return proc


def perf_stat(
    cmd: Sequence[str],
    events: Optional[Sequence[str]],
    out_path: Path,
    cwd: Optional[Path] = None,
) -> subprocess.Popen:
    """Start perf stat for the given command, redirect output to out_path.

    Does not set a duration; the perf exits when the command exits.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", buffering=1, encoding="utf-8")
    perf_cmd = ["perf", "stat"]
    if events:
        perf_cmd += ["-e", ",".join(events)]
    perf_cmd += ["--", *cmd]
    return subprocess.Popen(perf_cmd, stdout=fout, stderr=subprocess.STDOUT, text=True, cwd=str(cwd) if cwd else None)


def detect_siblings() -> Dict[int, str]:
    """Return mapping cpu -> thread_siblings_list string for convenience."""
    mapping: Dict[int, str] = {}
    base = Path("/sys/devices/system/cpu")
    for p in base.glob("cpu[0-9]*/topology/thread_siblings_list"):
        try:
            cpu = int(p.parts[-3][3:])  # 'cpuN'
            mapping[cpu] = p.read_text().strip()
        except Exception:
            continue
    return mapping
