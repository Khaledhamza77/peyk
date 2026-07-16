# Peyk

Private document parsing: layout detection → per-region digitalization (text / tables /
figures) → assembled Markdown. Self-hosted or your-own-cloud-account only — no public/anonymous
vendor APIs. Currently a containerized local pipeline (Phase 1 of the PoC); AWS cloud
deployment (API, queue, GPU worker) is planned but not yet built.

See [docs-personal/pipeline.md](docs-personal/pipeline.md) for the full pipeline design and
model rationale, [docs-personal/poc_architecture.md](docs-personal/poc_architecture.md) /
[poc_architecture.mmd](docs-personal/poc_architecture.mmd) for the target cloud architecture, and
[docs-personal/implementation_plan.md](docs-personal/implementation_plan.md) for detailed,
checkbox-level build status.

## Status

- **Phase 1 (containerization & local testing): in progress, core pipeline working end to end.**
  All nine containers below run and have been verified through real `peyk-orchestrator`
  dispatch against a sample document (`data/cib_sample.pdf`, Arabic financial statement).
  Remaining gaps: broader document-family coverage (only Family A tested so far), scanned-doc
  coverage (only born-digital tested, scanned simulated via a config override), and pushing
  images to ECR.
- **Phase 2-5 (CloudFormation IaC, AWS deployment, SDK/CLI, demo readiness): not started.**

## Pipeline

Two-step spine: **Layout Understanding** (detect + classify page regions) → **Digitalization**
(per-region: text, tables, figures) → **Markdown fragments**, concatenated using the layout
detector's own region order (a raster top-to-bottom/left-to-right sort — a lightweight
heuristic, not a full reading-order solver). Covers document Families A (structured
regulatory/financial), B (legal/contractual), and D (correspondence); Family C (transactional)
is deferred to its own pipeline.

## Containers

| Container | Role |
|---|---|
| `peyk-layout` | Layout Understanding — region detection + classification. Backends: **Heron** (default), PP-DocLayoutV2, DocLayout-YOLO, Surya. |
| `peyk-dcr` | Digital Character Recognition — direct text extraction from a born-digital PDF's text layer. No model. |
| `peyk-simple-ocr` | Self-hosted OCR, in-process backends: PaddleOCR, EasyOCR, RapidOCR, Tesseract. |
| `peyk-paddleocr-vl` | Thin HTTP client for the PaddleOCR-VL-0.9B backend (talks to `peyk-vllm-paddleocr`). |
| `peyk-vllm-paddleocr` | Persistent vLLM sidecar serving PaddleOCR-VL-0.9B (started once, not dispatched per-stage). |
| `peyk-surya` | Client for Surya-OCR-2, a single VLM covering layout/TSR/OCR via different prompts (talks to `peyk-vllm-surya`); also a standalone full-page transcription mode. |
| `peyk-vllm-surya` | Persistent vLLM sidecar serving Surya-OCR-2. |
| `peyk-tsr` | Table Structure Recognition, self-hosted. Backends: **TableFormer** (default), PP-StructureV3 (general/wiring), RapidTable, TATR, Surya. |
| `peyk-vlm` | The one container calling a managed LLM API — Bedrock (Claude, Nova) or Vertex AI in a private GCP project (Gemini, DeepSeek-OCR). Four roles: `ocr`, `figure` description, `table` (structure+text together), `fullpage` transcription. |
| `peyk-orchestrator` | Config-driven dispatcher — reads `config/example.yaml`, does the born-digital-vs-scanned check, dispatches the other containers per region/page, and assembles the final Markdown. |

## Model selection

`peyk-orchestrator` is entirely config-driven (`containers/peyk-orchestrator/config/example.yaml`)
— each pipeline job (`layout`, `tsr`, `ocr`, `cell_ocr`, `figures`, optionally `fullpage`) names
a `model`, and the orchestrator resolves which container/backend to dispatch. Current defaults:
layout → Heron, TSR → TableFormer, OCR → PaddleOCR-VL, figures → a Vertex Gemini model. Any
`peyk-vlm`-backed model (Bedrock or Vertex) can substitute into `ocr`/`tsr`/`figures`/`fullpage`
directly by name; run `docker run --rm peyk-vlm:dev --list-models` for the full live list.

## Privacy constraint

Self-hostable, or a managed LLM API running inside a private-cloud account boundary you control
— no public/anonymous vendor APIs. `peyk-vlm` supports both AWS Bedrock and Vertex AI (in a
private GCP project), with cross-cloud credentials held as real secrets, not hardcoded.

## Running locally

Requires Docker with GPU support (NVIDIA CUDA + WSL2 on Windows) for the GPU-backed containers.

1. Build each container under `containers/` (`docker build`).
2. Start the persistent OCR sidecars you need (`containers/peyk-vllm-paddleocr/start.sh` and/or
   `containers/peyk-vllm-surya/start.sh`).
3. Drop a PDF into `hotstorage/input/`.
4. Run `containers/peyk-orchestrator/run_local.sh` — dispatches the configured pipeline via
   docker-outside-of-docker and writes assembled Markdown to `hotstorage/output/`.

Config lives at `containers/peyk-orchestrator/config/example.yaml`; override with
`PEYK_CONFIG=<path>`. See that file's comments for every job's available models.

## License

See [LICENSE](LICENSE).
