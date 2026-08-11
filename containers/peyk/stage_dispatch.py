"""In-process stage dispatch — the merged image's replacement for docker-outside-of-docker's
`docker run <image> --stage <x> ...` (docs-personal/new_containerization_strategy.md, step 5).
Importable from any stage's own code as `import stage_dispatch`: /app (this file's own
directory) is always on sys.path — the Python interpreter puts a script's own directory there
at startup (ENTRYPOINT ["python3", "run.py"], WORKDIR /app) and nothing removes it afterward,
regardless of which stage_dir subsequent dispatches insert at sys.path[0].

Used by:
- run.py's own top-level --stage routing (the original single-dispatch-per-process case)
- stages/orchestrator/stages.py's run_docker_stage, calling sibling stages in-process instead
  of shelling out to `docker run`
- stages/surya/backends/tsr_dispatch.py, folding its own nested docker-in-docker call to the
  tsr stage into the same mechanism

Every stage's own `backends` package (and any other stage-local top-level module name) shares
the same name across stages, resolved only by whichever directory happens to be at the front of
sys.path at import time — safe when each dispatch was a brand-new OS process (the old model, one
`docker run` per stage, fresh sys.modules every time) but not once multiple stages run
sequentially inside the SAME long-lived process. _purge_stage_modules() evicts anything
previously loaded from under stages/ before each dispatch, so the next stage's own `backends`
resolves fresh instead of a stale entry left behind by whichever stage ran last. Third-party
packages (torch, paddlex, transformers, ...) are never touched by this — they live under
site-packages, not stages/, so they stay cached across dispatches within one process, which is
strictly a win over the old model (which paid full container-start + fresh-interpreter +
fresh-import cost on every single stage dispatch).

Dispatches are serialized process-wide via a reentrant lock: the orchestrator stage dispatches
its own sibling stages (layout, tsr, ocr, ...) synchronously and sequentially on one thread, so
those nested calls must be able to re-enter call_stage on the SAME thread without deadlocking
(hence RLock, not Lock) — but two dispatches from DIFFERENT threads (e.g. surya's own
ThreadPoolExecutor workers, each calling the tsr stage concurrently for a different table's
split-and-recognize fallback) must never actually run at once, since sys.path/sys.argv/
sys.modules mutation during dispatch is global, process-wide state with no per-thread isolation.
This trades away the old model's genuine OS-level parallelism for nested dispatches (previously
up to 2 truly-parallel `docker run` processes) for correctness (now serialized, one at a time) —
a real, deliberate tradeoff, not an oversight.

Each dispatch also releases any GPU memory the stage it just ran was holding (see
_release_gpu_memory) — the old model got this for free (a container exiting destroys its whole
CUDA context unconditionally), but in one shared process, a stage's model objects going out of
scope when main() returns does NOT free their GPU memory: PyTorch's caching allocator keeps
freed memory reserved for reuse within the SAME process rather than returning it to the driver.
Without this, GPU memory used by e.g. the layout stage's Heron model would stay reserved (fully
invisible from Python, but very visible in nvidia-smi) for the rest of this process's life, and
later stages would compete for headroom against memory nothing is actually using anymore.
"""
import gc
import runpy
import sys
import threading
from pathlib import Path

STAGES_ROOT = Path(__file__).parent / "stages"

_DISPATCH_LOCK = threading.RLock()


def _purge_stage_modules() -> None:
    root = str(STAGES_ROOT)
    stale = [name for name, mod in sys.modules.items() if str(getattr(mod, "__file__", "") or "").startswith(root)]
    for name in stale:
        del sys.modules[name]


def _release_gpu_memory() -> None:
    """gc.collect() first: a stage's model/tensor objects may only be reachable through
    reference cycles (common in complex third-party model code — tqdm handles, thread pools,
    framework-internal caches) that plain refcounting won't have freed yet when main() returns,
    and torch.cuda.empty_cache() can only return memory for tensors that have actually been
    deallocated. synchronize() before empty_cache(): if any of the stage's CUDA work is still
    queued/in-flight on the stream when main() returns (plausible — inference calls are
    asynchronous by default), empty_cache() can't reclaim memory tied to work that hasn't
    actually completed yet; blocking until the device is idle first closes that gap. Safe to
    call unconditionally: a stage that never imported torch (dcr, vlm, paddleocr-vl, the
    orchestrator itself) just hits the ImportError branch and no-ops."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def call_stage(stage: str, argv: list[str]) -> int:
    """Runs stages/<stage>/run.py's main() in-process, argv-equivalent to what the old
    `docker run <image> ... <argv>` call would have passed after the image name (argv here
    excludes argv[0] — the program name — same convention sys.argv[1:] would use). Returns an
    exit code the same way a real subprocess would have: main()'s own return value if it
    returns normally, or the code from a SystemExit if it calls sys.exit()/parser.error()
    instead (argparse's own error path) — either way, this never lets an uncaught SystemExit
    propagate up and kill the whole (possibly orchestrator-hosting) process over what should be
    just one failed stage dispatch."""
    stage_dir = STAGES_ROOT / stage
    with _DISPATCH_LOCK:
        _purge_stage_modules()
        sys.path[:] = [p for p in sys.path if not p.startswith(str(STAGES_ROOT))]
        sys.path.insert(0, str(stage_dir))
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *argv]
            main = runpy.run_path(str(stage_dir / "run.py"))["main"]
            try:
                return main()
            except SystemExit as exc:
                return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        finally:
            sys.argv = old_argv
            _release_gpu_memory()
