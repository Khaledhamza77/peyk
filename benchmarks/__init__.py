"""Latency and compute benchmarking for the peyk pipeline.

Measures what can be measured without ground truth -- wall time, per-stage time, GPU memory,
utilisation and power, normalised by the units of work actually performed. Accuracy is *cited*
from each model's published figures (model_cards.py), never measured here, because this project
has no annotated test set to measure against.

    from benchmarks import Harness, cases, report

    harness = Harness(input_dir="hotstorage/input", output_dir="hotstorage/output",
                      config_dir="hotstorage/peyk-config")
    harness.run_matrix(cases.build(["ocr"]), repeats=3)
    path = harness.save("benchmarks/results/ocr.json")
    print(report.render(path))
"""
from .harness import BenchmarkCase, Harness, RunRecord, StageTiming, aggregate, environment_fingerprint
from .gpu import GpuSampler, GpuTrace
from .workunits import WorkUnits, count as count_work_units

__all__ = [
    "BenchmarkCase",
    "Harness",
    "RunRecord",
    "StageTiming",
    "aggregate",
    "environment_fingerprint",
    "GpuSampler",
    "GpuTrace",
    "WorkUnits",
    "count_work_units",
]
