# Peyk PoC — Implementation Plan (AWS)

Tracks progress phase by phase. Each task is broken into unitary checkboxes — individual steps that must each be done for the task to be done. Check off a box only once its own verification passes, not when it merely looks complete.

Detailed findings, decisions, and gotchas are **not** here — see [build_notes.md](build_notes.md), organized by the same task numbers. This file only tracks what's done and what's left.

Reference: [pipeline.md](pipeline.md) (model candidates), [poc_architecture.mmd](poc_architecture.mmd) (architecture diagram).

Platform: **Amazon Web Services**, using a **$200** free-credit grant. Self-imposed spend cap: **$50** (well under the credit, to be safe). This is the second migration for this project — originally scoped for AWS, moved to GCP after account-signup verification issues, now moving back to AWS. Phase 0/2/3 below reset to not-done since they're cloud-specific; Phase 1 (container work) carries over as-is since it's cloud-agnostic. See [build_notes.md](build_notes.md) for the GCP-era history, kept rather than discarded.

---

## Phase 0 — Account & Foundations

### 0.1 Set up the AWS account

- [ ] AWS account confirmed usable, $200 free-credit grant linked/visible. Verify: `aws sts get-caller-identity` succeeds; credit balance visible in Billing console.

### 0.2 Secure the account, set up CLI access

- [ ] MFA enabled on the AWS root account.
- [ ] IAM user or role created for day-to-day/CLI work (not the root account).
- [ ] `aws configure` (or SSO login) completed locally. Verify: `aws sts get-caller-identity` shows the expected account ID and IAM identity, not root.

### 0.3 Set up billing guardrails

- [ ] AWS Budget created (`Peyk PoC Budget`, $50 USD) with alerts at 50%/80%/100% ($25/$40/$50). Verify: budget visible in Billing → Budgets, or via `aws budgets describe-budgets`.

### 0.4 Get managed-LLM access for OCR fallback / figure description

- [ ] Bedrock model access requested for **Claude Haiku 4.5**. See [pipeline.md](pipeline.md) access notes and [build_notes.md](build_notes.md#task-04--managed-llm-access-for-ocr-fallback--figure-description) for why this was dropped on GCP and why Bedrock is expected to be lower-friction.
- [ ] Verify: a live `InvokeModel`/`Converse` call against Claude Haiku 4.5 on Bedrock succeeds in the chosen region, with no purchase step or manual quota-increase request needed.

### 0.5 Create ECR repositories

- [ ] Four ECR repositories created: `peyk-layout`, `peyk-ocr`, `peyk-table`, `peyk-figure`, in a region with both `g4dn` GPU instance availability and Bedrock Claude Haiku 4.5 availability (default assumption: `us-east-1` — broadest AWS service/model availability; confirm before committing). Verify: `aws ecr describe-repositories --region <region>` shows all four.

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
- [ ] Surya-Layout backend — deferred to its own independent container (decision, not a gap).
- [ ] Coverage verification across Families A, B, D — only spot-tested on one document (`cib_sample.pdf`) so far.
- [ ] `--model` switch confirmed to actually change which backend runs — done for the three implemented backends.

### 1.3 Build `peyk-ocr` container

Pluggable `OCRBackend` interface + CLI (`run.py --model <backend> --lang <arabic|latin> --input <dir> --output <dir>`), takes a text-region crop, outputs recognized text. Build order and full rationale in [build_notes.md](build_notes.md#task-13--peyk-ocr-container).

- [X] `OCRBackend` interface + CLI scaffolded (`containers/peyk-ocr/`).
- [X] PaddleOCR backend (PP-OCRv5/v6, det+rec) — implemented, verified on GPU.
- [X] PaddleOCR-VL-0.9B backend — implemented, verified on GPU.
- [X] Granite-Docling-258M backend — implemented, verified on GPU (quality weakest of the six — see build notes).
- [X] EasyOCR backend — implemented, verified on GPU.
- [X] RapidOCR backend (PaddlePaddle engine) — implemented, verified on GPU.
- [X] Tesseract backend — implemented, verified on CPU (best quality of the six on the tested sample).
- [ ] Claude Haiku 4.5 backend (primary managed pick, via Bedrock) — not started, depends on Task 0.4 Bedrock access.
- [ ] Other managed Bedrock backends (Claude Sonnet 5/Opus 4.5, Amazon Nova Premier/Lite, DeepSeek-OCR, Mistral OCR) — not started, see [pipeline.md](pipeline.md) ranked list.
- [ ] AIN-7B backend — deferred to "Later" (decision, not a gap).
- [ ] Surya-recognition backend — deferred to "Later" (decision, not a gap).
- [ ] Verify all backends against a Latin-script crop — only Arabic tested so far.
- [ ] Verify all backends against a scanned-doc crop — only born-digital tested so far (`cib_sample.pdf`).

### 1.4 Build `peyk-table` container

- [ ] Containerize table structure recognition, pluggable across candidates: TableFormer, TATR, RapidTable, PP-StructureV3 (default TBD during build). Input: a table region; output: structured table (rows/cols/cells).
- [ ] Verify against one born-digital table from sample docs — row/column structure correct.
- [ ] Verify against one scanned table from sample docs — row/column structure correct.

### 1.5 Build `peyk-figure` container

- [ ] Containerize the figure/chart/stamp description step — calls Bedrock (Claude Haiku 4.5), no local GPU. Input: a figure-region crop; output: a text description.
- [ ] Verify against a chart from sample docs — plausible description returned.
- [ ] Verify against a stamp from sample docs — plausible description returned.
- [ ] Verify against a photo/figure from sample docs — plausible description returned.

### 1.6 Build born-digital vs. scanned detector

- [ ] Page-level check (interim granularity — target is per-region later): given a PDF page, returns `born-digital` or `scanned`.
- [ ] Verify against a known born-digital PDF — correct classification.
- [ ] Verify against a known scanned PDF (e.g. a flattened scan) — correct classification.

### 1.7 Build the local batch orchestrator

- [ ] Script that runs containers sequentially over a batch of jobs (mimicking the EC2 GPU instance's "one container loaded at a time" pattern): layout → born-digital check → (text/table/figure branches) → assemble markdown fragments.
- [ ] Verify against 2-3 sample docs per Family (A/B/D) — each doc produces markdown fragments covering all its text/table/figure regions.

### 1.8 End-to-end local validation

- [ ] Full pipeline run against a representative sample set covering Families A, B, D, both born-digital and scanned variants — no crashes.
- [ ] Manual review checklist per doc passes: layout regions look right, text is legible, tables are structured, figures have sensible descriptions.

### 1.9 Push images to ECR

- [ ] Tag and push all four validated container images to their ECR repositories, with a version tag (not just `latest`).
- [ ] Verify: fresh `docker pull` of each image (after local `docker rmi`, via `aws ecr get-login-password | docker login`) runs identically to the local build.

---

## Phase 2 — CloudFormation IaC

### 2.1 Storage config (S3)

- [ ] CloudFormation template: input-docs bucket and output-fragments bucket, Block Public Access enabled account-wide, default encryption (SSE-S3), no public bucket policies.
- [ ] Verify: `cfn-lint` passes; template review (or `cfn-guard`/`cfn-nag`) shows no public-access misconfigurations on either bucket.

### 2.2 Job state config (DynamoDB)

- [ ] CloudFormation template: DynamoDB table for job status (job id as partition key, status, timestamps, S3 pointers).
- [ ] Verify: `cfn-lint` passes.
- [ ] Verify (once deployed, Phase 3): a test item write/read succeeds.

### 2.3 Queue config (SQS)

- [ ] CloudFormation template: SQS queue for job submissions, a dead-letter queue, redrive policy configured.
- [ ] Verify: `cfn-lint` passes.
- [ ] Verify (once deployed, Phase 3): a test message sent and retrievable; failed messages route to the DLQ.

### 2.4 Submit API config (API Gateway + Lambda)

- [ ] CloudFormation template: API Gateway route + Lambda function accepting a doc upload, storing it in S3, creating a DynamoDB job record, sending an SQS message.
- [ ] Lambda execution role scoped to only the specific S3/DynamoDB/SQS resources it needs — no broad managed policies (e.g. `AmazonS3FullAccess`).
- [ ] Verify: `cfn-lint` passes; static review (or `cfn-guard`/`cfn-nag`) confirms no over-broad IAM grants on the function's role.

### 2.5 Status/result API config (API Gateway + Lambda)

- [ ] CloudFormation template: API Gateway route + Lambda reading job status from DynamoDB, returning an S3 presigned URL once complete.
- [ ] Least-privilege execution role: read-only on DynamoDB, presign-only on the output bucket.
- [ ] Verify: `cfn-lint` passes; static review confirms scoped permissions.

### 2.6 IAM review

- [ ] Consolidated pass over all IAM roles created across templates — no role has more permissions than its function needs, no primitive/broad managed policies (`AdministratorAccess`, `PowerUserAccess`) on functional roles.
- [ ] Verify: `cfn-guard`/`cfn-nag` findings resolved; cross-check with IAM Access Analyzer once deployed.

### 2.7 GPU compute config (EC2)

- [ ] CloudFormation template: EC2 instance (`g4dn.xlarge`) on AWS's Deep Learning AMI (GPU variant — NVIDIA drivers + Docker preinstalled), security group restricted to a known IP/CIDR only — no NAT Gateway.
- [ ] Instance role scoped to ECR pull + S3/SQS/DynamoDB access needed by the orchestrator.
- [ ] Verify: `cfn-lint` passes; security group confirmed to have no `0.0.0.0/0` ingress on management ports.

### 2.8 Bedrock access policy

- [ ] IAM policy on the EC2 instance role granting `bedrock:InvokeModel` scoped to the specific Claude Haiku 4.5 model ARN, not full Bedrock access.
- [ ] Verify: test invocation succeeds with this instance role; role has no unrelated permissions attached.

### 2.9 Full config validation pass

- [ ] `cfn-lint` across the full combined template set — zero errors.
- [ ] `cfn-guard`/`cfn-nag` (if available) across the full template set — zero high/critical findings.
- [ ] Re-run both after any template edit, before moving to Phase 3.

---

## Phase 3 — Deployment & Infra Testing

### 3.1 Deploy the stack

- [ ] CloudFormation stack deployed (`aws cloudformation deploy`/`create-stack`) to the AWS account — completes with no errors.
- [ ] Verify: `aws cloudformation describe-stacks` / resource listing confirms all expected resources exist.

### 3.2 Start the EC2 GPU instance, pull images

- [ ] Instance started manually, authenticated to ECR (`aws ecr get-login-password`), all four container images pulled.
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
- [ ] Final pick between RapidTable vs. TATR for table structure (pick during Task 1.4).
- [ ] EC2 lifecycle automation (auto-stop, scheduled start) — explicitly deferred; revisit after Phase 5.
- [ ] **EC2 GPU quota**: new/low-usage AWS accounts often start with a low or zero on-demand vCPU service quota for `g4dn`-family instances — request a quota increase (Service Quotas console) early in Phase 0, since approval can take time, mirroring the friction hit on both prior cloud attempts (original AWS signup, then GCP's T4 quota).
- [ ] `peyk-ocr` backend build order and rationale — see [build_notes.md](build_notes.md#task-13--peyk-ocr-container). The managed-LLM portion of that list is now Bedrock-based (Claude/Nova) rather than Agent Platform-based (Gemini) — see [pipeline.md](pipeline.md) for the updated ranked candidates.
