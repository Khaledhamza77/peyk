# Pipeline
Two-step spine: **Layout Understanding** (detect + classify regions) → **Digitalization** (per-element: text, tables, figures) → **Markdown fragments**. Assembly is limited to concatenating per-region fragments using the layout detector's own region ordering (PP-DocLayoutV2 and similar already emit a usable reading order) — no separate reading-order-solving component is built; this is not full document reconstruction. Covers document Families A (structured regulatory/financial), B (legal/contractual), D (correspondence) in one pipeline; Family C (transactional) deferred to its own pipeline later. Privacy constraint: self-hostable or private-cloud (**AWS Bedrock**) only — no public vendor APIs.

# Containers
- **`peyk-layout`** — Layout Understanding (region detection + classification).
- **`peyk-dcr`** — Digital Character Recognition: born-digital text, direct extraction from the source PDF's text layer. No model.
- **scanned text, self-hosted OCR only** (no managed-LLM calls live here — see `peyk-vlm`) — split across two containers plus one persistent sidecar (reformulated from a single `peyk-ocr`; see [implementation_plan.md](implementation_plan.md) Task 1.3 and [build_notes.md](build_notes.md#task-13--peyk-ocr-container) for why):
  - **`peyk-simple-ocr`** — pluggable in-process backends: PaddleOCR, EasyOCR, RapidOCR, Tesseract.
  - **`peyk-paddleocr-vl`** — thin HTTP client for the PaddleOCR-VL-0.9B backend, no local GPU/inference of its own.
  - **`peyk-vllm-paddleocr`** — persistent vLLM server `peyk-paddleocr-vl` talks to (started independently, not dispatched per-stage by `peyk-orchestrator` like the others — model load/CUDA-graph-capture cost is paid once rather than per document batch).
- **`peyk-tsr`** — Table Structure Recognition, self-hosted only. Always runs on a detected table region; paired with `peyk-simple-ocr`/`peyk-paddleocr-vl` (or `peyk-vlm`, if configured) when the table is scanned.
- **`peyk-vlm`** — the one container that calls Bedrock. Two roles, same underlying mechanism: (1) figure/chart/stamp description, and (2) OCR fallback or full replacement for scanned text, depending on config — the pipeline can be configured to call `peyk-vlm` only when the OCR stage fails, or as the primary/only text-recognition path for scanned regions, per deployment choice.
- **`peyk-orchestrator`** — lightweight, config-driven. Reads model-choice/param config, dispatches the other containers in sequence per region, does the born-digital-vs-scanned page check, and does final assembly (concatenate per-region markdown fragments using the layout detector's region order, format as output markdown).

# Ranked candidates
## Layout (`peyk-layout`):
PP-DocLayoutV2, Surya-Layout, DocLayout-YOLO, Heron, Detectron2

## OCR, self-hosted only (`peyk-simple-ocr` / `peyk-paddleocr-vl`):
PaddleOCR-VL-0.9B (`peyk-paddleocr-vl`, served via `peyk-vllm-paddleocr` — see Containers above), PaddleOCR (PP-OCRv5/v6), EasyOCR, Tesseract, RapidOCR (all four `peyk-simple-ocr`)
— AIN-7B, Surya-recognition: deferred to "Later," not yet built (see [implementation_plan.md](implementation_plan.md) Task 1.3).
— Granite-Docling-258M: dropped, not deferred — weakest quality of the originally-tried six, no longer a candidate.
— managed/Bedrock-backed text recognition (Claude, Amazon Nova, DeepSeek-OCR, Mistral OCR) is **not** a `peyk-simple-ocr`/`peyk-paddleocr-vl` backend; it lives in `peyk-vlm` instead (see below), so the same container/call path handles both figure description and OCR fallback/replacement.

## Table structure recognition (`peyk-tsr`):
TableFormer, Table Transformer (TATR), PP-StructureV3 (table module), RapidTable
— born-digital: TSR alone, no OCR pairing. Scanned: TSR + a model from the `peyk-simple-ocr`/`peyk-paddleocr-vl`/`peyk-vlm` text-recognition path.

## VLM — figure/chart/stamp description AND OCR fallback/replacement (`peyk-vlm`, ranked by cost):
Amazon Nova Lite, Claude Haiku 4.5, Amazon Nova Premier, Claude Sonnet 5, Claude Opus 4.5, DeepSeek-OCR (unverified — Bedrock access/request format not yet confirmed), Mistral OCR (unverified — access not yet confirmed on Bedrock)
— excluded: **Gemini family** (Google-first-party, not available on Bedrock — this was the GCP-era pick, excluded here purely by the platform move, not a quality judgment; see [build_notes.md](build_notes.md) for the GCP-era history), Qwen2.5-VL-7B/32B (no managed API on Bedrock, self-deploy only — needs an expensive dedicated GPU-backed endpoint, disproportionate to PoC scale)
— access notes: on GCP, Claude tiers were entitled via Agent Platform's global endpoint but blocked by a zero-default quota, and the Marketplace-purchase step required upgrading off trial billing — see [build_notes.md](build_notes.md) Task 0.4 for the full GCP-era account-friction history. On Bedrock, **confirmed working**: Claude Haiku 4.5 invoked successfully on the first real attempt, no purchase step, no quota wait (via the `eu.anthropic.claude-haiku-4-5-20251001-v1:0` cross-region inference profile — the bare model ID isn't directly invokable on-demand). Amazon Nova and the other Claude tiers not yet individually re-verified but expected to follow the same low-friction pattern.