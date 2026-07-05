# Peyk PoC — Implementation Plan (AWS)

Tracks progress phase by phase. Each task is broken into unitary checkboxes — individual steps that must each be done for the task to be done. Check off a box only once its own verification passes, not when it merely looks complete.

Detailed findings, decisions, and gotchas are **not** here — see [build_notes.md](build_notes.md), organized by the same task numbers. This file only tracks what's done and what's left.

Reference: [pipeline.md](pipeline.md) (model candidates), [poc_architecture.mmd](poc_architecture.mmd) (architecture diagram).

Platform: **Amazon Web Services**, account `615300991452`, region `eu-north-1`. Free-credit grant: **$100 confirmed, up to $100 more** unlockable via AWS's credit-earning activities (see Task 0.1). Self-imposed spend cap: **$50** (well under the confirmed credit, to be safe). This is the second migration for this project — originally scoped for AWS, moved to GCP after account-signup verification issues, now moving back to AWS. Phase 0/2/3 below reset to not-done since they're cloud-specific; Phase 1 (container work) carries over as-is since it's cloud-agnostic. See [build_notes.md](build_notes.md) for the GCP-era history, kept rather than discarded.

---

## Phase 0 — Account & Foundations

### 0.1 Set up the AWS account

- [X] AWS account confirmed usable. Verify: `aws sts get-caller-identity --profile peyk` succeeds (account `615300991452`).
- [X] Credit grant visible in Billing console — **$100 confirmed now**, up to $100 more unlockable via AWS's "Earn AWS credits" activities (launch an EC2 instance, use Bedrock playground, set up a cost budget, create a Lambda web app, create an Aurora/RDS database). Four of the five overlap directly with this plan; the RDS/Aurora one doesn't (this project uses DynamoDB) and is optional/bonus-only.

### 0.2 Secure the account, set up CLI access

- [X] MFA enabled on the AWS root account (confirmed via console).
- [X] IAM user created for day-to-day/CLI work (`peyk-cicd`, PowerUserAccess) — not the root account.
- [X] `aws configure --profile peyk` completed locally. Verify: `aws sts get-caller-identity --profile peyk` shows IAM user `peyk-cicd` in account `615300991452`, not root.

### 0.3 Set up billing guardrails

- [X] AWS Budget created (`My Monthly Cost Budget`, $50 USD) with alerts at 25/50%/70%/100% — in progress, being created via console (also completes one of the credit-earning activities). Verify: budget visible in Billing → Budgets, or via `aws budgets describe-budgets --profile peyk`.

### 0.4 Get managed-LLM access for OCR fallback / figure description

- [X] Bedrock model access confirmed for **Claude Haiku 4.5** in `eu-north-1` — no access request needed, worked immediately. See [build_notes.md](build_notes.md#task-04--managed-llm-access-for-ocr-fallback--figure-description) for the full comparison against the GCP-era friction.
- [X] Verify: live `InvokeModel` call against `eu.anthropic.claude-haiku-4-5-20251001-v1:0` (the cross-region inference profile ID — the bare model ID isn't directly invokable on-demand for this model) succeeded on the first real attempt.

---

## Phase 1 — Containerization & Local Testing

*(Cloud-agnostic — carries over unchanged from the GCP pass. Only the final push target and managed-LLM backend change.)*

### 1.1 Set up local GPU Docker environment

- [X] Docker Desktop + WSL2 + NVIDIA CUDA support working. Verify: `docker run --gpus all nvidia/cuda:12.9.2-base-ubuntu22.04 nvidia-smi` shows the local RTX 3500 Ada from inside a container.

### 1.2 Build `peyk-layout` container

Pluggable `LayoutBackend` interface + CLI (`run.py --model <backend> --input <dir> --output <dir>`), one container, multiple backends. See [build_notes.md](build_notes.md#task-12--peyk-layout-container).

- [X] `LayoutBackend` interface + CLI scaffolded (`containers/peyk-layout/`).
- [X] PP-DocLayoutV2 backend (default, via PaddleX) — implemented, verified on GPU against `data/cib_sample.pdf`.
- [X] DocLayout-YOLO backend (via `doclayout-yolo`) — implemented, verified on GPU against `data/cib_sample.pdf`.
- [X] Heron backend (Docling's layout model, via `docling-ibm-models`) — implemented, verified on GPU against `data/cib_sample.pdf`.
- [ ] **Bug found in Heron's raw output** (reported after initial verification above): `docling_ibm_models`'s `LayoutPredictor.predict()` only applies HF's `post_process_object_detection(..., threshold=...)` — a confidence-score filter, no NMS at all (DETR-family models are architecturally supposed to be NMS-free, but this one empirically isn't). Real output on `cib_sample.pdf` had duplicate/near-duplicate boxes, including identical bboxes under two different labels. Fixed in `backends/heron.py`: (1) raised `base_threshold` from the library default 0.3 to 0.5 to drop low-confidence spurious detections outright; (2) added class-agnostic NMS using **both** IoU and intersection-over-minimum-area (IoM) — plain IoU alone misses full-containment duplicates when box sizes differ a lot (an observed real case: a large box fully enclosing a smaller correct one scored IoU ≈ 0.12, far below any reasonable suppression threshold, because union is dominated by the large box). Rebuild + re-run against `cib_sample.pdf` to confirm not yet done.
- [ ] Coverage verification across Families A, B, D — only spot-tested on one document (`cib_sample.pdf`) so far.
- [X] `--model` switch confirmed to actually change which backend runs — done for the three implemented backends.
- [X] Heron and DocLayout-YOLO weights baked into the `peyk-layout` image at build time (`Dockerfile`), rather than downloaded at first run — found that every fresh `--rm` container was silently re-downloading both from HuggingFace Hub (`huggingface_hub.snapshot_download`/`hf_hub_download` caches to `~/.cache/huggingface`, a directory `peyk-orchestrator/stages.py` never mounted a persistent volume for, unlike `~/.paddlex`). Unlike PP-DocLayoutV2 (blocked from baking in — see `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` comment in the Dockerfile), neither of these two needs `paddlepaddle-gpu`'s GPU-dependent import, so nothing blocks a build-time download. Deliberately did **not** also mount a runtime cache volume for these two: doing so would shadow the baked-in image layer and silently defeat it.

### 1.3 Build OCR containers (reformulated — was a single `peyk-ocr`)

Originally one pluggable `OCRBackend` container. Split in a later session into **three**
pieces after the `paddleocr-vl` reliability investigation ([build_notes.md](build_notes.md#task-13--peyk-ocr-container))
concluded that model's local (in-process) inference path was structurally unfixable — see
that investigation for the full root-cause chain (naive token-by-token generate loop
syncing GPU→CPU every step, batch size hardcoded to 1, no way to pass a repetition-penalty
decode guard) and why serving it through vLLM instead resolves all three:

- **`peyk-simple-ocr`** — the pluggable-backend CLI (`run.py --model <backend> --lang <arabic|latin> --input <dir> --output <dir>`) survives here for every backend that runs
  in-process: PaddleOCR, EasyOCR, RapidOCR, Tesseract. Same interface and container
  lifecycle as before (per-stage `docker run --rm`, dispatched by `peyk-orchestrator`).
- **`peyk-paddleocr-vl`** — a thin, GPU-free HTTP client (no CUDA base image, no
  paddlepaddle-gpu; just `paddlex[genai-client]` + `openai`) that speaks OpenAI-compatible
  chat completions to `peyk-vllm-paddleocr` via PaddleX's built-in `engine="genai_client"` /
  `engine_config={"backend": "vllm-server", ...}` predictor path. Still dispatched by
  `peyk-orchestrator` as a normal per-stage `docker run --rm` container, same as any other
  stage.
- **`peyk-vllm-paddleocr`** — **not** a per-stage container. A persistent sidecar (started
  once via its own `start.sh`, left running, like a database) wrapping PaddlePaddle's
  official `paddleocr-genai-vllm-server` image, which bundles a vLLM build with
  PaddleOCR-VL's custom architecture (Ernie4.5 decoder + SigLIP-style vision encoder)
  already registered — mainline vLLM does not support this architecture out of the box, so
  this is not a stack we could easily build/maintain ourselves.

  **Why two containers instead of one merged client+server**, since this is the one place
  in the pipeline that breaks the "every stage is an ephemeral `docker run --rm`" pattern:
  vLLM engine startup (weight load, CUDA graph capture, KV cache allocation) is genuinely
  slow — merging the server into the per-stage ephemeral container would mean paying that
  cost on every document batch, which just relocates the original "10+ minutes for one crop"
  problem from decode-time to load-time instead of eliminating it. Splitting means that cost
  is paid once, the model stays warm across every stage invocation after that, and the
  per-stage container stays small (no CUDA/vLLM image weight) since it's just firing an HTTP
  request. The tradeoff this gives up: a standing GPU memory reservation
  (`vllm_config.yml`'s `gpu_memory_utilization`) for the server even when idle between runs —
  only worth it because this pipeline is exercised repeatedly (interactive/dev use), not as
  a rare nightly batch job.

- [X] `OCRBackend` interface + CLI scaffolded, now split across `containers/peyk-simple-ocr/`
  and `containers/peyk-paddleocr-vl/`.
- [X] PaddleOCR backend (PP-OCRv5/v6, det+rec) — implemented, verified on GPU (`peyk-simple-ocr`).
- [X] PaddleOCR-VL-0.9B backend — re-implemented as a `peyk-vllm-paddleocr` client
  (`peyk-paddleocr-vl`); local in-process path abandoned, see rationale above.
- [X] EasyOCR backend — implemented, verified on GPU (`peyk-simple-ocr`).
- [X] RapidOCR backend (PaddlePaddle engine) — implemented, verified on GPU (`peyk-simple-ocr`).
- [X] Tesseract backend — implemented, verified on CPU (`peyk-simple-ocr`; best quality of the
  original six on the tested sample).
- [X] Granite-Docling-258M backend — **dropped**, not deferred: weakest quality of the
  original six (see build notes) and added a torch/transformers/accelerate dependency
  footprint to `peyk-simple-ocr` for no corresponding benefit. Revisit only if a concrete
  gap shows up that the remaining backends don't cover.
- [ ] AIN-7B backend — deferred to "Later" (decision, not a gap).
- [ ] Surya-recognition backend — deferred to "Later" (decision, not a gap).
- [X] Verify `peyk-paddleocr-vl` end-to-end against a live `peyk-vllm-paddleocr` server —
  all 13 crops of the `cib_sample` test batch (`containers/peyk-orchestrator/_test/workdir/ocr_in/cib_sample`)
  processed correctly in 8.4s total (vs. the original single-crop 10+ minute pathological
  case), including `r9` (`هشام عز العرب`) — the exact crop that was low-confidence garbage
  on every backend previously tried (`paddleocr`, `tesseract`, and the old unreliable local
  `paddleocr-vl` path) — now transcribed correctly. `gpu_memory_utilization` had to be
  raised from an initial 0.35 to 0.7 in `peyk-vllm-paddleocr/vllm_config.yml`: vLLM's own
  torch.compile/CUDA-graph-capture overhead for this model already exceeded a 0.35 budget
  before any KV cache was allocated ("Available KV cache memory: -1.98 GiB").
- [X] `config.py` derives `ocr.image` (and, for `paddleocr-vl`, `ocr.server_url`) from
  `ocr.backend` automatically (`OCR_BACKEND_IMAGES`/`_ocr_stage`), instead of requiring both
  to be set in `config/example.yaml` and kept in sync by hand — verified: switching
  `backend: paddleocr-vl` → `backend: tesseract` in isolation correctly re-derives
  `peyk-paddleocr-vl:dev`+server URL → `peyk-simple-ocr:dev`+no server URL, with no other
  yaml changes needed.
- [X] Verify the same path through the actual `peyk-orchestrator` dispatch (`stages.py`/`pipeline.py`),
  not just a direct `docker run` against `peyk-paddleocr-vl` — ran the full pipeline against
  `cib_sample.pdf` end to end with both `backend: easyocr` and `backend: paddleocr-vl` in
  `config/example.yaml`; dispatched `docker run` commands confirmed correct image/backend/
  `--server-url` derivation and `--network peyk-net` join in both cases, output assembled
  correctly into `cib_sample.md`. `paddleocr-vl` run: `هشام عز العرب` (the target fix)
  transcribed correctly, and the bilingual `CB (CIB) البنك التجاري الدولي - مصر (سي أي بي) COMMERCIAL INTERNATIONAL BANK - EGYPT (CIB)` line handled correctly across scripts — but
  the documented hallucination risk reproduced live in this same run: a stray `一` (lone
  Chinese character) inserted between two unrelated lines, not present in the source. The
  vLLM migration fixed speed/reliability/non-determinism; it did not fix hallucination risk,
  which still carries no confidence signal — this remains a real, open quality caveat for
  `paddleocr-vl`, not a regression from this work.
- [ ] Re-verify `peyk-simple-ocr`'s four backends still behave identically post-split — code
  carried over unchanged from the original `peyk-ocr`, but not individually re-run since
  being relocated into the new container.
- [ ] Verify all backends against a Latin-script crop — only Arabic tested so far.
- [ ] Verify all backends against a scanned-doc crop — only born-digital tested so far (`cib_sample.pdf`); exercised via `peyk-orchestrator`'s `force_scanned` config override (routes born-digital regions through OCR anyway) as a stand-in, but that's not the same as a genuinely scanned/rasterized-only source document. See [build_notes.md](build_notes.md#task-13--peyk-ocr-container) for the extensive follow-up config tuning and the `paddleocr-vl` reliability investigation.

### 1.4 Build `peyk-dcr` container

New container (not part of the GCP-era plan) — see [pipeline.md](pipeline.md#containers). Takes a born-digital text region + source PDF, outputs directly-extracted text, no model involved.

- [ ] Scaffold container + CLI (mirroring the `run.py --input <dir> --output <dir>` pattern the other containers use, minus `--model` since there's only one approach).
- [ ] Direct text extraction implemented (e.g. via `pypdfium2`'s text-extraction API, already a dependency elsewhere in this project).
- [ ] Verify against a born-digital region from `data/cib_sample.pdf` — extracted text matches the source exactly (this should be much closer to guaranteed than any OCR backend, since there's no recognition step).

### 1.5 Build `peyk-tsr` container

*(Renamed from `peyk-table`.)*

- [ ] Containerize table structure recognition, pluggable across candidates: TableFormer, TATR, RapidTable, PP-StructureV3 (default TBD during build). Input: a table region; output: structured table (rows/cols/cells).
- [ ] Verify against one born-digital table from sample docs — row/column structure correct.
- [ ] Verify against one scanned table from sample docs — row/column structure correct.

### 1.6 Build `peyk-vlm` container

*(Renamed and expanded from `peyk-figure`.)* The one container that calls Bedrock — two roles, same call path: figure/chart/stamp description, and OCR fallback-or-full-replacement for scanned text (config-driven, see [pipeline.md](pipeline.md)). No local GPU.

- [ ] Containerize the figure/chart/stamp description step — calls Bedrock (Claude Haiku 4.5). Input: a figure-region crop; output: a text description.
- [ ] Containerize the OCR-fallback/replacement role — same Bedrock call path, input: a scanned text-region crop; output: recognized text. Config flag controls whether this only fires when `peyk-ocr` fails, or runs as the primary/only path.
- [ ] Verify figure role against a chart from sample docs — plausible description returned.
- [ ] Verify figure role against a stamp from sample docs — plausible description returned.
- [ ] Verify figure role against a photo/figure from sample docs — plausible description returned.
- [ ] Verify OCR role against a scanned text crop — recognized text matches source.

### 1.7 Build `peyk-orchestrator` container

*(Renamed and expanded from "the local batch orchestrator" — promoted from an implicit script to its own defined, lightweight, GPU-free container. Absorbs what was previously a standalone "born-digital vs. scanned detector" task.)* See [architecture_proposal.md](architecture_proposal.md) 4.7.

- [X] Reads a config of model choices/params (which backend each of `peyk-layout`/`peyk-ocr`/`peyk-tsr`/`peyk-vlm` should run, `peyk-vlm`'s OCR role mode) — implemented (`config.py`/`config/example.yaml`).
- [X] Born-digital vs. scanned page-level check implemented: given a PDF page, returns `born-digital` or `scanned` — implemented (`pipeline.py:born_digital_pages`), plus a `force_scanned` config override for testing the scanned branch against a born-digital sample doc. See [build_notes.md](build_notes.md#task-17--peyk-orchestrator-container).
- [X] Dispatches `peyk-layout` → (born-digital check) → `peyk-dcr`/`peyk-ocr` + `peyk-tsr` + `peyk-vlm` in sequence per document (mimicking the EC2 GPU instance's "one container loaded at a time" pattern) — implemented and verified for `peyk-layout`/`peyk-ocr` (real docker-outside-of-docker dispatch); `peyk-dcr`/`peyk-tsr`/`peyk-vlm` correctly stubbed pending Tasks 1.4/1.5/1.6.
- [X] Final assembly implemented: concatenates each region's markdown fragment using `peyk-layout`'s own region order into one output file per document (not a reading-order-solving component — see [pipeline.md](pipeline.md)) — implemented and verified, correct region order preserved end to end.
- [X] Verify born-digital check against a known born-digital PDF — correct classification (`cib_sample.pdf`, unforced check, correctly routed all text regions to the `peyk-dcr` stub rather than `peyk-ocr`).
- [ ] Verify born-digital check against a known scanned PDF (e.g. a flattened scan) — not yet done, no genuinely scanned sample doc available yet (only simulated via `force_scanned` on a born-digital doc).
- [ ] Verify full dispatch+assembly against 2-3 sample docs per Family (A/B/D) — only `cib_sample.pdf` (Family A) tested so far, extensively, but not multiple docs or other Families yet.
- [X] `stages.py`'s per-stage `docker run` no longer swallows the dispatched container's own stdout/stderr — `subprocess.run(cmd, capture_output=True, text=True)` was capturing it into Python variables and only surfacing it (in the exception message) on failure; silently discarded on success, so none of a stage's own logging (model loading, per-crop progress, etc.) was ever visible during a normal run. Fixed by dropping `capture_output` so it inherits and streams to the terminal live, same as running that `docker run` directly.

### 1.8 End-to-end local validation

- [ ] Full pipeline run against a representative sample set covering Families A, B, D, both born-digital and scanned variants — no crashes.
- [ ] Manual review checklist per doc passes: layout regions look right, text is legible, tables are structured, figures have sensible descriptions.

### 1.9 Push images to ECR

- [ ] Tag and push all seven validated container images (`peyk-layout`, `peyk-dcr`, `peyk-simple-ocr`, `peyk-paddleocr-vl`, `peyk-tsr`, `peyk-vlm`, `peyk-orchestrator`) to their ECR repositories, with a version tag (not just `latest`). `peyk-vllm-paddleocr` is not among these — see Task 1.3.
- [ ] Verify: fresh `docker pull` of each image (after local `docker rmi`, via `aws ecr get-login-password | docker login`) runs identically to the local build.

---

## Phase 2 — CloudFormation IaC

### 2.1 Create ECR repositories

*(Moved here from an earlier "jumps the queue" Task 0.5 — reformulated to live in its natural phase alongside the rest of the account's infra rather than as a standalone early exception.)*

- [ ] CloudFormation template written for ECR repositories: `peyk-layout`, `peyk-dcr`, `peyk-simple-ocr`, `peyk-paddleocr-vl`, `peyk-tsr`, `peyk-vlm`, `peyk-orchestrator` — seven, not the original six, now that `peyk-ocr` is split (see Task 1.3). `peyk-vllm-paddleocr` is **not** in this list: it's PaddlePaddle's prebuilt image run as a persistent sidecar, not something this project builds/pushes its own image for. Targeting `eu-north-1` (confirmed, not just assumed — Bedrock Claude Haiku 4.5 verified working there in Task 0.4; still need to confirm `g4dn` GPU availability in-region before fully committing, though `eu-north-1` virtually always has it).
- [ ] `cfn-lint` passes on the template, as part of the full Phase 2 set (Task 2.10), not reviewed standalone.
- [ ] Template deployed as part of Phase 2/3's normal write-then-deploy sequencing (Task 1.9, which needs these repos to push images to, now follows this task rather than preceding it).
- [ ] Verify: `aws ecr describe-repositories --region eu-north-1 --profile peyk` shows all seven.

### 2.2 Storage config (S3)

- [ ] CloudFormation template: input-docs bucket and output-fragments bucket, Block Public Access enabled account-wide, default encryption (SSE-S3), no public bucket policies.
- [ ] Verify: `cfn-lint` passes; template review (or `cfn-guard`/`cfn-nag`) shows no public-access misconfigurations on either bucket.

### 2.3 Job state config (DynamoDB)

- [ ] CloudFormation template: DynamoDB table for job status (job id as partition key, status, timestamps, S3 pointers).
- [ ] Verify: `cfn-lint` passes.
- [ ] Verify (once deployed, Phase 3): a test item write/read succeeds.

### 2.4 Queue config (SQS)

- [ ] CloudFormation template: SQS queue for job submissions, a dead-letter queue, redrive policy configured.
- [ ] Verify: `cfn-lint` passes.
- [ ] Verify (once deployed, Phase 3): a test message sent and retrievable; failed messages route to the DLQ.

### 2.5 Submit API config (API Gateway + Lambda)

- [ ] CloudFormation template: API Gateway route + Lambda function accepting a doc upload, storing it in S3, creating a DynamoDB job record, sending an SQS message.
- [ ] Lambda execution role scoped to only the specific S3/DynamoDB/SQS resources it needs — no broad managed policies (e.g. `AmazonS3FullAccess`).
- [ ] Verify: `cfn-lint` passes; static review (or `cfn-guard`/`cfn-nag`) confirms no over-broad IAM grants on the function's role.

### 2.6 Status/result API config (API Gateway + Lambda)

- [ ] CloudFormation template: API Gateway route + Lambda reading job status from DynamoDB, returning an S3 presigned URL once complete.
- [ ] Least-privilege execution role: read-only on DynamoDB, presign-only on the output bucket.
- [ ] Verify: `cfn-lint` passes; static review confirms scoped permissions.

### 2.7 IAM review

- [ ] Consolidated pass over all IAM roles created across templates — no role has more permissions than its function needs, no primitive/broad managed policies (`AdministratorAccess`, `PowerUserAccess`) on functional roles.
- [ ] Verify: `cfn-guard`/`cfn-nag` findings resolved; cross-check with IAM Access Analyzer once deployed.

### 2.8 GPU compute config (EC2)

- [ ] CloudFormation template: EC2 instance (`g4dn.xlarge`) on AWS's Deep Learning AMI (GPU variant — NVIDIA drivers + Docker preinstalled), security group restricted to a known IP/CIDR only — no NAT Gateway.
- [ ] Instance role scoped to ECR pull + S3/SQS/DynamoDB access needed by the orchestrator.
- [ ] Verify: `cfn-lint` passes; security group confirmed to have no `0.0.0.0/0` ingress on management ports.

### 2.9 Bedrock access policy

- [ ] IAM policy on the EC2 instance role granting `bedrock:InvokeModel` scoped to the specific Claude Haiku 4.5 model ARN, not full Bedrock access.
- [ ] Verify: test invocation succeeds with this instance role; role has no unrelated permissions attached.

### 2.10 Full config validation pass

- [ ] `cfn-lint` across the full combined template set — zero errors.
- [ ] `cfn-guard`/`cfn-nag` (if available) across the full template set — zero high/critical findings.
- [ ] Re-run both after any template edit, before moving to Phase 3.

---

## Phase 3 — Deployment & Infra Testing

### 3.1 Deploy the stack

- [ ] CloudFormation stack deployed (`aws cloudformation deploy`/`create-stack`) to the AWS account — completes with no errors.
- [ ] Verify: `aws cloudformation describe-stacks` / resource listing confirms all expected resources exist.

### 3.2 Start the EC2 GPU instance, pull images

- [ ] Instance started manually, authenticated to ECR (`aws ecr get-login-password`), all six container images pulled.
- [ ] Verify: `docker run --gpus all <image> nvidia-smi` (or equivalent) succeeds for GPU-using containers on the instance itself.

### 3.3 End-to-end infra path test

- [ ] One real document submitted through the full path: SDK/curl → API Gateway → Lambda → S3 + SQS → EC2 orchestrator picks it up → processes → writes output → status Lambda reports complete.
- [ ] Verify: submitted doc reaches `status=complete`, presigned URL returns valid markdown fragments.
- [ ] Verify: output content matches Phase 1's local run for the same document (parity check).
- [ ] Time the full round trip.

### 3.4 Cost check-in

- [ ] AWS Billing/Cost Explorer + Budgets dashboard checked after first end-to-end test.
- [ ] Verify: spend in line with earlier estimate, well within the $200 credit.
- [ ] Verify: billing report grouped by service shows only expected services (EC2, S3, Bedrock, Lambda, DynamoDB, SQS, API Gateway, ECR) — no surprise line items (e.g. an accidental NAT Gateway).

---

## Phase 4 — SDK & CLI

### 4.1 SDK: `submit()`

- [ ] `submit(file_path) -> job_id` implemented, wraps the submit API Gateway endpoint.
- [ ] Verify: real file submitted, job appears in DynamoDB with matching ID and `status=queued`.

### 4.2 SDK: `get_status()`

- [ ] `get_status(job_id) -> status` implemented, returns one of `queued | processing | complete | failed`.
- [ ] Verify: polling a real in-flight job shows correct status transitions through its lifecycle.

### 4.3 SDK: `get_result()`

- [ ] `get_result(job_id) -> markdown fragments` implemented, returns actual content (not just the presigned URL) for a completed job.
- [ ] Verify: fetched result for a known completed job diffed against expected output from Task 3.3.

### 4.4 CLI wrapper

- [ ] `peyk submit <file>`, `peyk status <job_id>`, `peyk fetch <job_id>` implemented on top of the SDK.
- [ ] Verify: all three subcommands run in sequence against a fresh document, end to end, from a clean shell, with no additional setup beyond AWS credentials/API endpoint config.

---

## Phase 5 — End-to-End Testing & Demo Readiness

### 5.1 Sample doc sweep

- [ ] Representative set of documents from Families A, B, D (mix of born-digital and scanned) run through the CLI/SDK — all process successfully.
- [ ] Markdown fragment quality reviewed and documented (known good/bad cases noted).
- [ ] Verify checklist per document: correct region detection, legible text, structured tables, sensible figure descriptions.

### 5.2 Latency/concurrency sanity check

- [ ] A handful of jobs (matching expected demo user count) submitted concurrently — all complete without errors.
- [ ] Verify: no jobs stuck in `queued`/`processing` indefinitely; single EC2 orchestrator handles the queue depth.
- [ ] Total time for the batch to clear the queue noted.

### 5.3 Demo dry run

- [ ] Full walkthrough as if presenting to company users — start EC2 instance, submit via CLI/SDK, show result, stop instance — completes with no manual intervention beyond documented start/stop steps.
- [ ] Verify: a second person (not the builder) follows a short runbook and reproduces the demo unaided.

---

## Open Questions / Decisions Deferred

- [ ] Confirm whether this AWS pass uses a fresh account or the original AWS account from before the GCP migration — affects whether prior signup-verification issues are already resolved.
- [ ] Final pick between DocLayout-YOLO vs. PP-DocLayoutV2 for layout default (both viable; pick during Task 1.2 based on quickest integration).
- [ ] Final pick between RapidTable vs. TATR for table structure (pick during Task 1.5).
- [ ] EC2 lifecycle automation (auto-stop, scheduled start) — explicitly deferred; revisit after Phase 5.
- [ ] **EC2 GPU quota**: new/low-usage AWS accounts often start with a low or zero on-demand vCPU service quota for `g4dn`-family instances — request a quota increase (Service Quotas console) early in Phase 0, since approval can take time, mirroring the friction hit on both prior cloud attempts (original AWS signup, then GCP's T4 quota).
- [ ] `peyk-ocr` backend build order and rationale — see [build_notes.md](build_notes.md#task-13--peyk-ocr-container). The managed-LLM portion of that list is now Bedrock-based (Claude/Nova) rather than Agent Platform-based (Gemini) — see [pipeline.md](pipeline.md) for the updated ranked candidates.
