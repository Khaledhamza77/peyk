"""GPU resource sampling for a benchmark run.

Polls `nvidia-smi` in a background thread for the duration of one pipeline run and keeps the raw
time series, so a stage's resource cost can be recovered afterwards by slicing that series against
the stage's own start/end timestamps (see harness.py's per-stage attribution). Deliberately
subprocess-based rather than pynvml: nvidia-smi ships with the driver and is already present on
any machine that can run this pipeline at all, whereas pynvml is an extra dependency not
currently in the SDK's pins.

One-shot polling per sample rather than `nvidia-smi --loop-ms`: a long-lived streaming child has
to be reaped correctly on every abnormal exit path (timeout, KeyboardInterrupt, a run that
raises), and getting that wrong leaves a stray nvidia-smi running against the same GPU the
benchmark is trying to measure. Per-sample spawn cost (~50-100ms on Windows) is irrelevant
against runs measured in minutes.

IMPORTANT CAVEAT this module cannot solve for you, and which report.py surfaces explicitly:
memory.used is whole-board, not per-process. When a vLLM sidecar is up it has already reserved
its share (peyk-vllm-surya runs at --gpu-memory-utilization 0.85-0.90), and that reservation is
included in every sample taken while it lives. Quote peak against the baseline captured before
the workload started -- vram_peak_delta_mib below -- and read the absolute number as "total board
occupancy", never as "what this stage needed".
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

# Fields pulled per sample, in this order -- _query() zips them positionally onto GpuSample.
_QUERY_FIELDS = (
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
)

_NVIDIA_SMI = shutil.which("nvidia-smi")

# nvidia-smi reports these per-field on hardware/driver combinations that do not expose the
# metric -- power.draw in particular is unavailable on many laptop GPUs. Treated as a missing
# value for that one field rather than a failed sample, so a card without power telemetry still
# yields usable memory/utilization traces.
_MISSING = ("[N/A]", "[Not Supported]", "[Unknown Error]", "N/A")


@dataclass(frozen=True)
class GpuSample:
    t: float
    memory_used_mib: float
    memory_total_mib: float
    util_gpu_pct: float | None
    util_mem_pct: float | None
    power_w: float | None
    temp_c: float | None


@dataclass
class GpuTrace:
    """Raw samples plus the pre-run baseline. Every aggregate is computed on demand rather than
    accumulated during sampling, so one trace can be re-sliced by arbitrary time windows after
    the fact -- which is exactly what per-stage attribution needs."""

    samples: list[GpuSample] = field(default_factory=list)
    baseline_mib: float | None = None
    gpu_name: str | None = None
    available: bool = True
    # Set when sampling was requested but nvidia-smi is missing or failed, so report.py can print
    # "not measured" rather than zeros that would be indistinguishable from a real measurement.
    unavailable_reason: str | None = None

    def window(self, start: float, end: float) -> "GpuTrace":
        return GpuTrace(
            samples=[s for s in self.samples if start <= s.t <= end],
            baseline_mib=self.baseline_mib,
            gpu_name=self.gpu_name,
            available=self.available,
            unavailable_reason=self.unavailable_reason,
        )

    def summary(self) -> dict:
        if not self.available:
            return {"gpu": self.gpu_name, "available": False, "reason": self.unavailable_reason}
        if not self.samples:
            # A stage that finished faster than one sampling interval genuinely has no samples.
            # Reported as such rather than as 0 MiB, which would read as "measured, used nothing".
            return {
                "gpu": self.gpu_name,
                "available": True,
                "samples": 0,
                "note": "no samples in window (stage shorter than sampling interval)",
            }

        mem = [s.memory_used_mib for s in self.samples]
        util = [s.util_gpu_pct for s in self.samples if s.util_gpu_pct is not None]
        power = [s.power_w for s in self.samples if s.power_w is not None]
        temp = [s.temp_c for s in self.samples if s.temp_c is not None]

        out: dict = {
            "gpu": self.gpu_name,
            "available": True,
            "samples": len(self.samples),
            "vram_peak_mib": round(max(mem), 1),
            "vram_mean_mib": round(sum(mem) / len(mem), 1),
            "vram_total_mib": round(self.samples[0].memory_total_mib, 1),
        }
        if self.baseline_mib is not None:
            # The number to actually quote for "what did this cost": whole-board usage minus
            # whatever was already resident (sidecars, plus the desktop compositor on a laptop
            # GPU that also drives the display -- true of this project's own dev card).
            out["vram_peak_delta_mib"] = round(max(mem) - self.baseline_mib, 1)
            out["vram_baseline_mib"] = round(self.baseline_mib, 1)
        if util:
            out["util_gpu_mean_pct"] = round(sum(util) / len(util), 1)
            out["util_gpu_peak_pct"] = round(max(util), 1)
        if power:
            out["power_mean_w"] = round(sum(power) / len(power), 1)
            out["power_peak_w"] = round(max(power), 1)
            # Mean power over the sampled span -- a rough energy figure, honest to roughly the
            # sampling interval. Fine for "GPU watt-hours per page" cost framing, not for
            # anything needing real power-metering accuracy.
            span = self.samples[-1].t - self.samples[0].t
            if span > 0:
                out["energy_wh"] = round((sum(power) / len(power)) * span / 3600.0, 4)
        if temp:
            out["temp_peak_c"] = round(max(temp), 1)
        return out


def _parse_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw in _MISSING:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _run_smi(args: list[str]) -> str | None:
    if _NVIDIA_SMI is None:
        return None
    try:
        proc = subprocess.run(
            [_NVIDIA_SMI, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _query(gpu_index: int) -> GpuSample | None:
    stdout = _run_smi(
        [
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
    )
    if not stdout or not stdout.strip():
        return None
    values = [_parse_float(part) for part in stdout.strip().splitlines()[0].split(",")]
    # memory.used/memory.total are the two fields everything else is reported against; a sample
    # missing either is unusable rather than partially usable.
    if len(values) != len(_QUERY_FIELDS) or values[0] is None or values[1] is None:
        return None
    return GpuSample(
        t=time.time(),
        memory_used_mib=values[0],
        memory_total_mib=values[1],
        util_gpu_pct=values[2],
        util_mem_pct=values[3],
        power_w=values[4],
        temp_c=values[5],
    )


def gpu_name(gpu_index: int = 0) -> str | None:
    stdout = _run_smi(["--query-gpu=name", "--format=csv,noheader", "-i", str(gpu_index)])
    return stdout.strip() if stdout else None


def driver_version() -> str | None:
    stdout = _run_smi(["--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"])
    return stdout.strip() if stdout else None


class GpuSampler:
    """Context manager. Captures a baseline sample on entry -- before the workload starts, but
    deliberately *after* any sidecars are already up, so a persistent vLLM reservation lands in
    the baseline instead of being misattributed to the run being measured."""

    def __init__(self, interval_s: float = 1.0, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.trace = GpuTrace()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        if _NVIDIA_SMI is None:
            self.trace.available = False
            self.trace.unavailable_reason = "nvidia-smi not found on PATH"
            return self
        baseline = _query(self.gpu_index)
        if baseline is None:
            self.trace.available = False
            self.trace.unavailable_reason = "nvidia-smi present but returned no usable sample"
            return self
        self.trace.gpu_name = gpu_name(self.gpu_index)
        self.trace.baseline_mib = baseline.memory_used_mib
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-sampler")
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = _query(self.gpu_index)
            if sample is not None:
                self.trace.samples.append(sample)
            # wait() rather than sleep() so teardown returns promptly instead of blocking for a
            # full interval.
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
        return False
