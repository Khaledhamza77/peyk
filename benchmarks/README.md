# Benchmarking peyk

Strategy, metric definitions, and the measurement hazards found while building this. Written to
be the source material for a technical presentation — every claim here is either measured, cited,
or explicitly labelled as an unverified observation.

## 1. The evidence contract

peyk orchestrates third-party models; it trains none of them. There is **no ground-truth corpus
and no annotated test set** in this project. That single fact determines what may honestly be
claimed, and it splits the evidence into three classes that must never be blended:

| Class | What it covers | Where it comes from | Needs ground truth? |
|---|---|---|---|
| **Measured** | Latency, per-stage time, GPU memory/utilisation/power, work-unit counts | This harness, on your own hardware | No — this is why it is the part worth measuring |
| **Cited** | Recognition accuracy per model | Each model's own published benchmark figures, with a source URL | Yes — which is why we cite rather than measure |
| **Shown** | Output quality on real documents | The demo's own side-by-side examples | No — qualitative, not scored |

Presenting a cited number as if it were measured here is the one failure mode this whole design
exists to prevent. `model_cards.py` carries the citations; `report.py` prints the disclaimer above
every published-accuracy table.

## 2. What is measured, and in what units

Raw seconds-per-document is close to meaningless across a mixed corpus: a 3-page document with one
120-cell table costs more than a 20-page prose document. Every latency figure is therefore
normalised by the units of work actually performed.

| Metric | Unit | Denominator | Notes |
|---|---|---|---|
| `wall_s` | seconds | — | Whole job, container start to exit |
| per-stage `duration_s` | seconds | see below | From the orchestrator's own `dispatch_end` events |
| `layout` normalised | s/page | rendered pages | Layout runs once per page |
| `ocr` normalised | s/region crop | whole-region crops | See the merge caveat in §4 |
| `cell_ocr` normalised | s/table cell | cell crops | Only when cells get their own dispatch |
| `tsr` / `table_full` | s/table | table regions | |
| `figures` | s/figure | figure regions | Managed VLM — network-bound |
| `dcr` | s/document | documents | No model; PDF text-layer extraction |
| `vram_peak_delta_mib` | MiB | — | Whole-board peak minus pre-run baseline. **Read §4.1 before quoting.** |
| `util_gpu_mean_pct` | % | — | Mean over the run's sampling window |
| `power_mean_w` / `energy_wh` | W / Wh | — | Absent on cards without power telemetry |

Work-unit counts come from the workdir volume, scoped to the directories *this job's own dispatch
events* touched — see §4.2.

### Cost, for managed models

Not yet captured. `VLMResult` (`containers/peyk/stages/vlm/backends/base.py`) carries only `text`
and a hardcoded `score=1.0`; the Bedrock and Vertex responses both return token usage and it is
currently discarded. Adding `input_tokens`/`output_tokens` there is the single highest-value
instrumentation change remaining, because for VLM-heavy configurations cost per page — not
latency — is usually the deciding number.

## 3. Experiment design

**Stage-isolated sweeps, never a cross-product.** layout(4) × tsr(5) × ocr(6) × cell_ocr × figures
is thousands of runs that answer no question anyone asked. Each sweep in `cases.py` freezes every
slot but one, so any difference is attributable to the one thing that changed. That is ~23 runs
total instead of thousands.

Available sweeps: `layout`, `ocr`, `tsr`, `fullpage`, `smart-split`, `born-digital`.

**`fullpage` is the control arm.** It bypasses layout/TSR/OCR entirely — one model call per page.
It is what tells you whether the per-region pipeline's complexity buys anything.

**Two ablations worth their own slide:**
- `smart-split` — the Surya blurred-table correction on/off. This is the one knob whose *quality*
  effect the project has genuinely characterised (7 tables, 3 documents), so pairing that recorded
  observation with its measured latency cost is defensible.
- `born-digital` — `force_scanned` on/off. The live config sets it `true`, routing born-digital
  pages through OCR/VLM even though the DCR text-layer path would extract them with no model, no
  GPU, and no API call. This measures exactly what that setting costs.

**Warmup and repeats.** One warmup run per case, recorded but never aggregated; then N measured
repeats. Results report **median plus min/max**, not mean and stddev — at realistic repeat counts
(3–5) a mean is one outlier away from being wrong, and for managed APIs an outlier is a real
recurring event (provider-side throttling), not noise to average away.

## 4. Measurement hazards

Every item below was found by running this, not by reasoning about it. They are the reason the
numbers are trustworthy, and several make good slides in their own right.

### 4.1 Whole-board VRAM cannot isolate the pipeline

`nvidia-smi` reports whole-board memory. Per-process attribution needs permissions the dev machine
does not grant — `--query-compute-apps` returns `[Insufficient Permissions]`. On the RTX 3500 Ada
(12,282 MiB) used here, **10,121 MiB was already held by unrelated processes before any run
started**, leaving peyk's own delta at roughly 685–1137 MiB.

Two runs happened to capture their baseline while that external allocation was absent (1,837 and
2,687 MiB), producing a "9,043 MiB" delta for a configuration whose *peak* matched every other
run. The harness now excludes runs whose baseline disagrees with the case median by more than
250 MiB, and prints `unstable` rather than a fabricated figure when too few agree.

**Consequence for a slide:** quote VRAM only from a machine with a quiet GPU, and quote the
baseline alongside it. Never present a delta without saying what the board was already holding.

### 4.2 The workdir volume accumulates across runs

`/hotstorage/workdir` is deliberately persistent — it is what makes intermediate crops inspectable
between runs. So it retains directories from earlier runs under different configs. Counting the
whole volume folds those into the current run: an early count reported 40 pages, 5 documents and
32 fullpage images for a 2-page single-document run that never used fullpage mode. Work units are
now scoped to the directories the job's own dispatch events name.

### 4.3 `ocr` and `cell_ocr` merge into one dispatch

`pipeline.py` gives table cells their own dispatch only when `cell_ocr` differs from `ocr`; when
the two resolve to equal configs it merges them to avoid loading the same local model twice. So
the `ocr` stage's duration means different things in different rows of the same table. Measured
case: 7 region crops vs 364 cells — dividing the merged duration by crops alone overstated
per-crop cost ~50×. Reports now use a combined **Text rec** column and switch the denominator to
`crops + cells` when the merge is detected.

### 4.4 Born-digital documents bypass OCR entirely

With `force_scanned: false`, text on born-digital pages goes through the DCR text-layer path and
**no OCR stage runs at all** — a first OCR sweep produced 0 region crops and no `ocr` event, so
all six backends would have tied at zero seconds. Every sweep except the `born-digital` ablation
now sets `force_scanned: true`, which also matches the config the project actually runs.

### 4.5 Sidecars must not be restarted between runs

`SidecarManager.start()` force-removes and recreates its container. Calling `ensure_sidecars()`
per run re-pays the full cold start every repeat — roughly 14 minutes for Surya on this card —
making the warmup pointless and every measured repeat a cold-start measurement. Sidecars are now
started once and held across repeats and across consecutive cases needing the same set, and always
outside the sampling window: their load cost is a one-time deployment cost, not per-document
latency. Their VRAM reservation still lands in the sampler's baseline.

### 4.6 Batch dispatch hides per-document latency

TSR/OCR/figures each dispatch **once for the whole batch**, not per document. For per-document
figures, run one PDF per invocation; for throughput figures, run the full batch. Both are valid
and they answer different questions — but they are not the same number.

### 4.7 Small inputs are fixed-cost dominated

At 2 pages, `layout` measured ~2.3 s/page, but that is dominated by model load rather than
per-page inference. Use a larger input before quoting any per-unit figure.

## 5. What is cited, and the caveat that matters most

`model_cards.py` holds every published figure with its exact benchmark, version, and source URL.
Headline examples: TableFormer 96.75% TEDS on PubTabNet; DocLayout-YOLO 78.8% mAP on
DocStructBench; PaddleOCR-VL 92.86 overall on OmniDocBench v1.5; Surya-OCR-2 83.3% on olmOCR-bench.

**These are not comparable to each other** — a TEDS, an mAP, and an olmOCR-bench score measure
different things on different data.

**And nearly all of them are dominated by English/Chinese born-digital pages, while this corpus is
predominantly Arabic, right-to-left, and substantially scanned.** The breakdowns that actually
speak to this use case deserve far more weight than the headline numbers:

- **PaddleOCR-VL**: Arabic accuracy 80.45, Arabic edit distance 0.122 — the strongest published
  Arabic figures in the set.
- **Surya-OCR-2**: Base 99.7% but **Old Scans 41.8%** — the vendor's own weakest category by a
  wide margin.

Both independently corroborate findings this project reached on its own: Surya's full-table path
was observed failing on genuinely scanned tables, and PaddleOCR-VL was observed to have the
highest quality ceiling of any backend tried.

For the managed VLMs (Claude, Gemini, Nova, Pixtral) there is **no vendor-published
document-parsing benchmark** comparable to OmniDocBench — they publish document-VQA-class results
instead. Recording that absence is more honest than substituting a third-party leaderboard number
and presenting it alongside vendor-authored ones.

## 6. Observations from this project (small-n, not benchmarks)

State the sample size inline. These are directional findings that motivated the current defaults;
an engineering audience will discount an unqualified ranking the moment they ask how it was
produced, and will accept the same ranking readily when it arrives already labelled.

| Observation | Evidence base |
|---|---|
| Tesseract strongest on Arabic; both crops byte-exact | n=2 crops, 1 born-digital document |
| PaddleOCR-VL highest ceiling, but hallucinated unprompted content | n=2 crops |
| TSR ranking: TableFormer > pp-structure-general > rapidtable > tatr | born-digital tables, 1 document |
| Isolated table-cell crops recognise badly on VLMs generally | reproduced on 2 independent VLMs |
| Blurred-table split triggers on Laplacian variance, not table size | 7 tables, 3 documents |
| One bold Arabic name failed on every backend tried | 1 crop, 6 backends; confidence 0.36–0.68 flagged it correctly |
| Bedrock crop test: Sonnet/Opus byte-exact; Nova Lite, Nova Pro, Pixtral hallucinated | n=1 crop |
| Vertex Gemini: 7/7 byte-exact or near-exact on the same crop | n=1 crop |

That last confidence-score result is itself the useful finding: the pipeline can *detect* the
failure even though no backend can fix it.

## 7. Running it

```bash
python -m benchmarks run --sweep ocr --repeats 3
python -m benchmarks run --sweep layout ocr tsr --repeats 3 --report report.md
python -m benchmarks report --results benchmarks/results/ocr.json
python -m benchmarks cards          # published figures only; no Docker or GPU needed
```

Useful flags: `--input`, `--output`, `--config-dir`, `--image`, `--sample-interval`, `--no-warmup`.

Credentials are picked up automatically from `containers/peyk/.env` and
`containers/peyk/gcp-key.json`, the same way `demo.py` does. They are needed even for
"self-hosted" sweeps, because the reference config still routes `figures` to a VLM — there is no
self-hosted figures backend.

### Before you run: the image must be current

The SDK now passes `--job-id` to the orchestrator (part of the uncommitted traceability work).
**The published `peyk:dev` image predates that flag**, so every SDK-driven run fails immediately
with `run.py: error: unrecognized arguments: --job-id`. Rebuild before benchmarking:

```bash
docker build -t peyk:dev containers/peyk
```

A full rebuild re-runs the paddle wheel install from Baidu's CDN, which timed out here. If that
happens, layer the changed source over the existing image instead — seconds, and equivalent for
benchmarking purposes:

```bash
printf 'FROM peyk:dev\nCOPY stages/orchestrator stages/orchestrator\nCOPY stages/vlm stages/vlm\nCOPY stage_dispatch.py .\nCOPY run.py .\n' \
  | docker build -t peyk:bench -f - containers/peyk
python -m benchmarks run --sweep ocr --image peyk:bench
```

## 8. Worked results

Real runs on this machine — RTX 3500 Ada Laptop (12,282 MiB), one 2-page Arabic document
(`cib_sample.pdf`), `force_scanned: true`, 2 measured repeats after a discarded warmup. Small
input, so treat per-page figures as fixed-cost dominated (§4.7); the *relative* ordering is the
signal.

**OCR backend sweep** (layout=heron, tsr=tableformer, cell_ocr=paddleocr):

| Configuration | Median (s) | Range (s) | Text rec (s) | VRAM delta (MiB) |
|---|---|---|---|---|
| ocr=paddleocr | 24.3 | 24.2–24.4 | 8.8 | unstable |
| ocr=tesseract | 24.7 | 23.8–25.6 | 9.4 | 758 |
| ocr=easyocr | 27.0 | 26.7–27.3 | 11.9 | 1047 |
| ocr=rapidocr | 45.2 | 45.1–45.3 | 29.7 | 1042 |

RapidOCR is a clear outlier at ~3× the recognition cost of the others — consistent with the
box-splitting inconsistency already recorded for it in `build_notes.md`.

**Layout backend sweep** (stage time only, the fair comparison):

| Backend | Layout stage (s) | VRAM delta (MiB) |
|---|---|---|
| doclayout-yolo | 3.4 | 685 |
| heron | 4.4 | 758 |
| pp-doclayout-v2 | 6.0 | 1027 |

Repeatability was tight throughout (e.g. 45.1 / 45.3 s), so a 3-run median is sufficient for
self-hosted backends. Managed backends will need more repeats — their variance is external.

## 9. Limits, stated plainly

- **No accuracy is measured here, and none can be** without a gold set.
- VRAM is whole-board, not per-process, and unreliable on a shared GPU (§4.1).
- Managed-model **cost is not captured** at all — token usage is currently discarded.
- Results are specific to this GPU, image, and commit; all three are recorded in every result file.
- Reading order is not evaluated. Assembly is a raster sort, known-wrong on multi-column layouts;
  nothing here measures that.

### If you later want real accuracy numbers

The cheapest credible path is a small gold set — 30–50 pages spanning born-digital vs scanned,
Arabic vs Latin, table-heavy vs prose. Bootstrap it by running the strongest available config
(a `fullpage` pass with a top-tier VLM), then hand-correcting the output. That is far cheaper than
transcribing Arabic financial tables from scratch. With that in place the natural metrics are CER
and WER for text (applying `normalize_digits` plus Arabic NFKC/tatweel/diacritic normalisation to
*both* sides before scoring), TEDS-Struct for tables, mAP for layout, and Kendall's τ against a
reference region order for reading order.
