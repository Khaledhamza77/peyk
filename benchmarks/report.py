"""Turns a saved benchmark run into slide-ready tables.

Reads the JSON written by Harness.save() -- never re-runs anything. Report definitions change far
more often than it is worth re-paying GPU time for, so scoring is deliberately offline and
re-runnable against a stored result set.

Output is Markdown because that is what pastes cleanly into slides, docs, and the deck generator
alike. Every table carries its own caveat line; those lines are not decoration. A latency table
without "measured on this GPU, this image, this commit" is not reproducible, and a model-card
table without "published by the vendor, not measured here" invites exactly the misreading this
whole benchmark is built to avoid.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import model_cards
from .harness import RunRecord, StageTiming, aggregate, failures
from .workunits import WorkUnits


def _units_from(raw: dict) -> WorkUnits | None:
    """Rebuilds a WorkUnits from a stored record so its denominator() logic is reused here rather
    than duplicated. Unknown keys are dropped so an older result file, written before a counter
    was added, still renders instead of raising."""
    if not raw:
        return None
    fields = {f for f in WorkUnits.__dataclass_fields__}
    return WorkUnits(**{k: v for k, v in raw.items() if k in fields})


def _load(path: str | Path) -> tuple[dict, list[RunRecord]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for raw in payload.get("records", []):
        stages = [StageTiming(**s) for s in raw.pop("stages", [])]
        records.append(RunRecord(stages=stages, **raw))
    return payload.get("environment", {}), records


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out) + "\n"


def environment_block(env: dict) -> str:
    dirty = " (uncommitted changes present)" if env.get("git_dirty") else ""
    lines = [
        "## Measurement environment",
        "",
        f"- **GPU**: {env.get('gpu') or 'not detected'} (driver {env.get('gpu_driver') or 'n/a'})",
        f"- **Host**: {env.get('host') or 'n/a'}",
        f"- **Commit**: {env.get('git_sha') or 'n/a'}{dirty}",
        f"- **Image**: {', '.join(env.get('image_tags') or []) or 'n/a'}",
        f"- **Captured**: {env.get('captured_at') or 'n/a'}",
        "",
        "Every latency figure below is specific to this hardware and this build. A single "
        "consumer laptop GPU that also drives the display is not a deployment target -- read "
        "these numbers as *relative* comparisons between models, never as absolute throughput.",
        "",
    ]
    return "\n".join(lines)


def metric_glossary() -> str:
    """Plain-language definitions for every column this report prints, meant to stand alone in
    front of someone who hasn't read benchmarks/README.md. Placed first, before any table, since
    a table of unexplained numbers invites the reader to guess at what they mean rather than ask."""
    return (
        "## What these numbers mean\n\n"
        "**\"Document batch\":** every timed run processes all sample PDFs in the input directory "
        "together, in a single pipeline invocation -- not one document at a time. The pipeline is "
        "built this way on purpose (layout/OCR/TSR/figures are each dispatched once per batch, not "
        "once per document, so each stage's model-load cost is paid once rather than repeated per "
        "document). This run's batch is the project's 5 sample PDFs, ~40 pages total, one of them "
        "a 30-page document that dominates the total. A row's Median therefore describes "
        "\"all 5 documents together,\" never \"one typical document\" -- dividing by page count "
        "(the s/page column, where available) is what makes a single-document cost comparable.\n\n"
        "- **Configuration** -- which pipeline setting was swept for that row (e.g. `ocr=tesseract` "
        "means every other stage stayed fixed and only the OCR backend changed to Tesseract).\n"
        "- **n** -- how many measured repeats that row's numbers are built from. A separate, "
        "discarded warmup run happens first and is never counted here (it pays one-time costs -- "
        "model loading, sidecar startup -- that are not part of steady-state latency).\n"
        "- **Median (s)** -- the middle value of those n repeats' total wall-clock time for one "
        "full pipeline run over the whole document batch (see above). Median, not mean/average, "
        "on purpose: with only a handful of repeats a single slow outlier (a network hiccup on a "
        "managed API call, for instance) would drag a mean far from what actually typically "
        "happens.\n"
        "- **Range (s)** -- the fastest and slowest of those repeats. A wide range is itself a "
        "finding -- it means that configuration's cost is unpredictable, not just slow.\n"
        "- **Text rec (s)** -- combined OCR + table-cell-OCR time. Reported together because the "
        "pipeline sometimes merges those two into a single dispatch internally, so the OCR figure "
        "alone would not mean the same thing in every row.\n"
        "- **s/page** -- median time divided by pages processed, so a 2-page run and a 30-page run "
        "can be compared on the same footing. `n/a` means the page count for that row was not "
        "captured (true for every row in a partial/live snapshot -- see the disclaimer below).\n"
        "- **VRAM delta (MiB)** -- extra GPU memory used, above whatever was already resident "
        "before the run started. `unstable` means something else on the machine changed how much "
        "memory it was holding between repeats, so no honest number could be computed -- not a "
        "failed measurement, a withheld one. `n/a` means VRAM was not sampled at all for that row.\n"
        "- **GPU util** -- average percent of the GPU actually busy during the run. Low util with "
        "high latency usually means the bottleneck is a network call (a managed API) or CPU work, "
        "not the GPU.\n"
        "- **Per-stage breakdown** (Stage / Model / Median (s) / Share) -- the same total run time "
        "split by pipeline stage (layout, ocr, tsr, figures, ...), so \"this configuration is "
        "slower\" becomes \"...because its OCR stage is\". Share is that stage's percent of the "
        "row's own total.\n\n"
    )


def partial_snapshot_disclaimer(completed: int, total: int, pending: list[str]) -> str:
    """What has and has not actually run yet, stated plainly, first, before any number. A partial
    result presented without this reads as a finished comparison -- the single most likely way
    this snapshot gets misquoted in a meeting."""
    lines = [
        "## Disclaimer -- this sweep was not finished when this was captured",
        "",
        f"**{completed} of {total} planned configurations** have at least one completed measured "
        "run below. The rest were still queued or in progress and are listed at the bottom under "
        "\"Not yet completed\" -- their absence here means *not yet measured*, not *found to be "
        "unnecessary* or *ruled out*.",
        "",
        "Additionally, because this is a snapshot of an in-progress run rather than a finished "
        "one's own saved output, **GPU memory, GPU utilization, and per-page normalisation are not "
        "available** for any row -- that data only ever lives in the process actually running the "
        "sweep, and could not be recovered independently. Every number below is wall-clock latency "
        "only.",
        "",
    ]
    if pending:
        lines.append(f"**Still pending ({len(pending)}):** " + ", ".join(pending))
        lines.append("")
    return "\n".join(lines)


def latency_table(summary: dict[str, dict]) -> str:
    """One row per swept case. Median plus range, because with 3-5 repeats a mean is one outlier
    away from being wrong -- and for managed APIs, outliers are a real recurring behaviour
    (provider-side throttling), not noise to be averaged away."""
    rows = []
    for case, entry in sorted(summary.items(), key=lambda kv: kv[1]["wall_s_median"]):
        units = entry.get("units", {})
        pages = units.get("pages") or 0
        per_page = f"{entry['wall_s_median'] / pages:.2f}" if pages else "n/a"
        vram = entry.get("vram_peak_delta_mib_median")
        util = entry.get("util_gpu_mean_pct_median")
        # Combined text-recognition cost. Reported instead of the bare `ocr` stage because
        # pipeline.py merges cell_ocr into ocr whenever the two configs are equal -- so the `ocr`
        # row alone means different things in different rows of this very table, while
        # ocr + cell_ocr always means the same thing.
        stages = entry.get("stages", {})
        text_rec = sum(stages[s]["median_s"] for s in ("ocr", "cell_ocr") if s in stages)
        rows.append([
            case,
            entry["n"],
            f"{entry['wall_s_median']:.1f}",
            f"{entry['wall_s_min']:.1f}-{entry['wall_s_max']:.1f}",
            f"{text_rec:.1f}" if text_rec else "-",
            per_page,
            f"{vram:.0f}" if vram is not None else ("unstable" if entry.get("vram_unstable") else "n/a"),
            f"{util:.0f}%" if util is not None else "n/a",
        ])
    body = _table(
        ["Configuration", "n", "Median (s)", "Range (s)", "Text rec (s)", "s/page",
         "VRAM delta (MiB)", "GPU util"],
        rows,
    )
    caveat = (
        "\n_**Text rec (s)** is `ocr` + `cell_ocr` combined, and is the column to compare when "
        "sweeping recognition backends: the orchestrator merges the two into one dispatch "
        "whenever their configs are equal, so the `ocr` stage alone is not the same quantity in "
        "every row._\n\n"
        "_**VRAM delta** is peak whole-board usage minus the pre-run baseline, for the whole run "
        "-- it is not attributable to the swept stage. Every configuration here still runs heron "
        "and tableformer on the GPU, so even a CPU-only recognition backend shows a non-zero "
        "figure. Per-stage attribution is in the breakdown below. Sidecar reservations sit in the "
        "baseline, not in this column, because sampling starts after they are already resident._\n\n"
        "_`unstable` means the GPU baseline moved between this case's runs by more than the "
        "tolerance, so no honest delta can be computed. nvidia-smi reports whole-board usage and "
        "per-process attribution is unavailable without elevated permissions, so a delta is only "
        "meaningful when nothing else on the machine claims or releases VRAM mid-sweep. Close "
        "other GPU applications and re-run rather than quoting a number from this column._\n"
    )
    return "## Latency and compute (measured)\n\n" + body + caveat


def failures_block(records: list[RunRecord]) -> str:
    """Configurations where every measured run failed. Without this, such a case simply has no
    row anywhere in the report -- indistinguishable from a configuration nobody tried. A
    configuration that reliably fails on this corpus is itself a real result."""
    failed = failures(records)
    if not failed:
        return ""
    sections = ["## Failed configurations", "", "Every measured run failed identically. Not a "
                "gap in coverage -- these were attempted and did not produce usable output.", ""]
    for case, info in sorted(failed.items()):
        sections.append(f"### {case}\n")
        sections.append(f"_{info['attempts']} attempt(s), all failed._\n")
        for err in info["errors"]:
            # Long tracebacks are fenced rather than inlined so they read as pre-formatted
            # detail, not as prose the reader is expected to parse line by line.
            sections.append(f"```\n{err}\n```\n")
    return "\n".join(sections)


def stage_breakdown_table(summary: dict[str, dict]) -> str:
    """Where the time actually goes, per case. This is usually the most useful slide in the deck:
    it is what turns "config A is slower" into "config A is slower *because* its OCR stage is"."""
    sections = []
    for case, entry in sorted(summary.items()):
        rows = []
        units = _units_from(entry.get("units", {}))
        present = set(entry["stages"])
        total = sum(s["median_s"] for s in entry["stages"].values()) or 1.0
        for stage, stats in sorted(entry["stages"].items(), key=lambda kv: -kv[1]["median_s"]):
            # Per-unit cost is the number that actually compares across configurations: a stage
            # that took longer because it was handed more crops is not a slower stage.
            denom = units.denominator(stage, present) if units else None
            if denom is not None:
                unit_name, n = denom
                per_unit = f"{stats['median_s'] / n:.3f} s/{unit_name} (n={n})"
            else:
                per_unit = "-"
            rows.append([
                stage,
                stats["model"] or "-",
                f"{stats['median_s']:.1f}",
                f"{100 * stats['median_s'] / total:.0f}%",
                per_unit,
            ])
        if rows:
            sections.append(
                f"### {case}\n\n"
                + _table(["Stage", "Model", "Median (s)", "Share", "Normalised"], rows)
            )
    if not sections:
        return ""
    caveat = (
        "\n_Normalised cost divides each stage's time by the units it was actually handed -- "
        "pages for layout, region crops for OCR, cells for cell_ocr, tables for TSR. Raw seconds "
        "are not comparable across documents: one 120-cell table costs more than twenty prose "
        "pages._\n"
    )
    return "## Per-stage breakdown (measured)\n\n" + "\n".join(sections) + caveat


def published_accuracy_table(model_keys: list[str] | None = None) -> str:
    """Vendor-published figures only. The header says so, every row carries its benchmark name,
    and the source list follows -- so nothing here can be mistaken for something peyk measured."""
    cards = (
        [c for c in model_cards.ALL_CARDS if c.key in set(model_keys)]
        if model_keys
        else list(model_cards.ALL_CARDS)
    )
    rows = []
    for card in cards:
        if not card.claims:
            rows.append([card.display, card.params or "-", "no published parsing benchmark", "-", "-"])
            continue
        for i, claim in enumerate(card.claims):
            rows.append([
                card.display if i == 0 else "",
                (card.params or "-") if i == 0 else "",
                claim.benchmark,
                claim.metric,
                claim.value,
            ])
    header = (
        "## Published accuracy (reported by each model's authors -- NOT measured here)\n\n"
        "peyk orchestrates third-party models rather than training any, and this project has no "
        "ground-truth corpus or annotated test set. Reporting a measured accuracy would therefore "
        "mean inventing one. What follows is what each model's own authors published, cited to "
        "source.\n\n"
        "**These numbers are not comparable to each other.** A TEDS on PubTabNet, an mAP on "
        "DocLayNet, and an olmOCR-bench score measure different things on different data. Compare "
        "within a row group, never across.\n\n"
    )
    return header + _table(["Model", "Params", "Benchmark", "Metric", "Reported"], rows)


def relevance_caveat_block() -> str:
    """The single most important slide in the accuracy half of the deck. Every headline benchmark
    above is dominated by English/Chinese born-digital pages; this corpus is neither."""
    lines = [
        "## Why the published numbers overstate the case here",
        "",
        "Nearly every benchmark cited above is dominated by **English and Chinese, born-digital** "
        "pages. This pipeline's documents are predominantly **Arabic, right-to-left, and "
        "substantially scanned**. A high headline score is weak evidence for this corpus, so the "
        "few breakdowns that do speak to it deserve disproportionate weight:",
        "",
    ]
    rows = []
    for card in model_cards.ALL_CARDS:
        for claim in card.subscores:
            if any(t in claim.benchmark.lower() for t in ("arabic", "scan", "table", "multi-column")):
                rows.append([card.display, claim.benchmark, claim.metric, claim.value, claim.note or "-"])
    lines.append(_table(["Model", "Breakdown", "Metric", "Reported", "Note"], rows))
    lines.append(
        "\nTwo of these corroborate findings this project reached independently: Surya's own "
        "weakest published category is Old Scans (41.8% against 99.7% on clean pages), which "
        "matches what was observed when its full-table path was run on genuinely scanned tables; "
        "and PaddleOCR-VL's Arabic-specific figures are the strongest published Arabic numbers in "
        "the set, which matches it having had the highest observed quality ceiling here.\n"
    )
    return "\n".join(lines)


def observations_block() -> str:
    """This project's own recorded observations, carried at their real sample size.

    These are honest, useful, and small. Stating n inline is what keeps them defensible: an
    engineering audience will discount an unqualified ranking built on two crops the moment they
    ask how it was produced, and will accept the same ranking readily when it arrives already
    labelled."""
    return (
        "## Observed in this project (small-n, not a benchmark)\n\n"
        "Recorded during development against this repo's own sample PDFs. Sample sizes are stated "
        "because they are small -- these are directional observations that motivated the current "
        "defaults, not measurements.\n\n"
        + _table(
            ["Observation", "Evidence base"],
            [
                ["Tesseract was the strongest Arabic backend tried; both crops byte-exact",
                 "n=2 crops, 1 born-digital document"],
                ["PaddleOCR-VL had the highest quality ceiling but hallucinated unprompted content",
                 "n=2 crops; hallucination seen on an otherwise-correct run"],
                ["TSR ranking: TableFormer > pp-structure-general > rapidtable > tatr",
                 "born-digital tables, 1 document"],
                ["Isolated table-cell crops recognise badly on VLMs generally",
                 "reproduced on 2 independent VLMs (paddleocr-vl, surya)"],
                ["Blurred-table split triggers on Laplacian variance, not table size",
                 "7 tables across 3 documents"],
                ["One bold Arabic name failed on every backend tried",
                 "1 crop, all 6 OCR backends; confidence 0.36-0.68 flagged it correctly"],
            ],
        )
        + "\nThe confidence scores flagging that last failure correctly is itself the useful "
        "result: the pipeline can detect the failure even though no backend can fix it.\n"
    )


def sources_block() -> str:
    lines = ["## Sources", ""]
    lines += [f"- {url}" for url in model_cards.SOURCES]
    return "\n".join(lines) + "\n"


def render(results_path: str | Path, include_published: bool = True) -> str:
    env, records = _load(results_path)
    summary = aggregate(records)
    parts = [
        "# peyk benchmark report",
        "",
        metric_glossary(),
        environment_block(env),
        latency_table(summary),
        stage_breakdown_table(summary),
        failures_block(records),
    ]
    if include_published:
        parts += [
            published_accuracy_table(),
            relevance_caveat_block(),
            observations_block(),
            sources_block(),
        ]
    return "\n".join(p for p in parts if p)


def write(results_path: str | Path, out_path: str | Path) -> Path:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(results_path), encoding="utf-8")
    return out_path
