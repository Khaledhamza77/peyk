# Peyk

Private document parsing: layout detection → per-region digitalization (text / tables /
figures) → assembled Markdown. Self-hosted or your-own-cloud-account only — no public/anonymous
vendor APIs. Currently a containerized local pipeline (Phase 1 of the PoC); AWS cloud
deployment (API, queue, GPU worker) is planned but not yet built.

See [docs-personal/pipeline.md](docs-personal/pipeline.md) for the full pipeline design and
model rationale, [docs-personal/poc_architecture.md](docs-personal/poc_architecture.md) /
[poc_architecture.mmd](docs-personal/poc_architecture.mmd) for the target cloud architecture,
[docs-personal/implementation_plan.md](docs-personal/implementation_plan.md) for detailed,
checkbox-level build status, and
[docs-personal/new_containerization_strategy.md](docs-personal/new_containerization_strategy.md)
for the 10 → 3 image consolidation that produced the current container layout below.

## Status

- **Phase 1 (containerization & local testing): in progress, core pipeline working end to end.**
  The pipeline runs as 3 containers (below) with no docker-outside-of-docker — `peyk` dispatches
  every stage in-process and calls out to the two persistent vLLM sidecars over HTTP. Verified
  against 5 sample PDFs spanning born-digital, scanned, Latin, and Arabic documents. Remaining
  gaps: broader document-family coverage (only Family A tested so far), and pushing images to
  ECR.
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
| `peyk` | Merged worker + orchestrator. One `--stage` flag picks which role runs: `layout`, `tsr`, `ocr`, `dcr`, `vlm`, `surya`, `paddleocr-vl`, or `orchestrator` (the default) — every stage dispatches in-process, no docker-outside-of-docker. See [containers/peyk/README.md](containers/peyk/README.md). |
| `peyk-vllm-surya` | Persistent vLLM sidecar serving Surya-OCR-2 (layout/TSR/OCR via a single VLM, plus full-table and full-page transcription). |
| `peyk-vllm-paddleocr` | Persistent vLLM sidecar serving PaddleOCR-VL-0.9B. |

Originally 10 separate images (one per stage, wired together via docker-outside-of-docker); see
`docs-personal/new_containerization_strategy.md` for the full consolidation history, including
why the two vLLM sidecars are kept separate rather than merged further (investigated and
decided against — different vendor images/vLLM builds, no compelling value beyond disk usage).

## Model selection

`peyk`'s orchestrator stage is entirely config-driven
(`containers/peyk/stages/orchestrator/config/example.yaml`) — each pipeline job (`layout`,
`tsr`, `ocr`, `cell_ocr`, `figures`, optionally `fullpage`) names a `model`, and the orchestrator
resolves which backend to dispatch. Current defaults: layout → Heron, TSR → TableFormer, OCR →
PaddleOCR-VL, figures → a Vertex Gemini model. Any `vlm`-backed model (Bedrock or Vertex) can
substitute into `ocr`/`tsr`/`figures`/`fullpage` directly by name; run
`docker run --rm peyk:dev --stage vlm --list-models` for the full live list.

## Privacy constraint

Self-hostable, or a managed LLM API running inside a private-cloud account boundary you control
— no public/anonymous vendor APIs. `peyk`'s `vlm` stage supports both AWS Bedrock and Vertex AI
(in a private GCP project), with cross-cloud credentials held as real secrets, not hardcoded.

## Running locally

Requires Docker with GPU support (NVIDIA CUDA + WSL2 on Windows).

1. Build the merged worker image: `docker build -t peyk:dev containers/peyk`.
2. Start the persistent vLLM sidecars you need (`containers/peyk-vllm-paddleocr/start.sh`
   and/or `containers/peyk-vllm-surya/start.sh`).
3. Drop a PDF into `hotstorage/input/`.
4. Run `containers/peyk/run_local.sh` — dispatches the configured pipeline (one container, no
   docker socket) and writes assembled Markdown to `hotstorage/output/`.

Config lives at `containers/peyk/stages/orchestrator/config/example.yaml`; override with
`PEYK_CONFIG=<path>`. See that file's comments for every job's available models.

## License

See [LICENSE](LICENSE).
