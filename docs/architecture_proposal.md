# Peyk — PoC Architecture Proposal (AWS)

## 1. Summary

Peyk wraps the document-parsing pipeline defined in [pipeline.md](pipeline.md) — Layout Understanding → Digitalization → Markdown fragments — in an internal API/SDK so company users can call it programmatically. This is a **proof of concept**: the design optimizes for minimal cost and fast iteration at low, trusted, internal usage, not for scale, high availability, or defense against adversarial input. Section 7 makes explicit which security and reliability practices are intentionally skipped for that reason, and what would need to change before this graduates beyond a demo.

Platform: **Amazon Web Services**, running against a **$200** free-credit grant, with a self-imposed **$50** spend cap. (Second migration: originally scoped for AWS, moved to GCP after account-signup verification issues, now moving back to AWS — see [build_notes.md](build_notes.md) for the GCP-era history and why it's kept rather than discarded. Diagram: [poc_architecture.mmd](poc_architecture.mmd).)

Task breakdown: [implementation_plan.md](implementation_plan.md).

## 2. Goals & Constraints

- **Privacy constraint (hard)**: self-hostable or private-cloud (managed LLM API within the cloud provider's boundary) only — no public vendor APIs. This shaped every model choice below; on AWS, this is satisfied by **Amazon Bedrock** rather than Agent Platform/Vertex AI.
- **Scope**: document Families A (structured regulatory/financial), B (legal/contractual), D (correspondence). Family C deferred. Assembly/reading-order is out of scope — output is markdown fragments per element, not a reassembled document.
- **Cost**: PoC runs against a **$200** AWS free-credit grant, self-capped at **$50** — a firm ceiling to stay well under, not a target to spend down. An AWS Budget alert is configured at 50%/80%/100% of this cap ($25/$40/$50).
- **Scale**: a handful of internal users, async/batch usage pattern (submit → poll/fetch later, not synchronous request/response).
- **Team**: greenfield AWS account, serverless-first comfort, no existing infra to integrate with.

## 3. Architecture Overview

Request path: SDK/CLI → Amazon API Gateway → AWS Lambda (submit) → S3 (input) + DynamoDB (job record) + SQS (queue) → a manually-started EC2 GPU instance drains the queue and runs the pipeline stages sequentially → results land in S3 (output) → SDK/CLI polls a second Lambda (status/result) → signed URL (S3 presigned URL) once complete.

The serverless front end (API Gateway, Lambda, S3, DynamoDB, SQS) is always available and costs near-zero at rest. The GPU compute is the only piece with a meaningful idle cost, so it's isolated behind a queue and started/stopped independently of the API's availability — submissions always succeed and simply wait in the queue until the instance is up.

## 4. Key Decisions & Rationale

### 4.1 Async/batch over synchronous API
Given a handful of users and no latency SLA, a submit-then-poll pattern avoids needing an always-warm compute tier. This is the single biggest cost lever in the design — it's what makes "start the GPU instance only when needed" possible at all. Provider-agnostic; unchanged by the GCP→AWS move.

### 4.2 Two-step pipeline, sequential per-stage GPU containers
Following pipeline.md's spine (Layout Understanding → Digitalization), each stage (layout, OCR, table structure, figure/VLM description) runs as its own container, loaded onto the GPU one at a time via a batch orchestrator rather than all models resident simultaneously.

**Why**: the alternative — loading all models concurrently — creates real VRAM contention risk. AIN-7B alone is ~14GB at fp16, close to the ceiling of even a 16GB T4, let alone the local dev card (12GB). Sequential loading means every stage gets the full GPU to itself, removes the need for careful memory budgeting or quantization tradeoffs across models, and costs nothing extra since the workload is already async/batch (a few seconds of container swap time is immaterial).

### 4.3 Model selection: self-hosted vs. managed LLM, per element type
| Element | Approach | Rationale |
|---|---|---|
| Layout | Self-hosted, pluggable across PP-DocLayoutV2, Surya-Layout, DocLayout-YOLO, Heron (Detectron2 optional) | No managed equivalent exists; these are lightweight detectors, cheap to self-host. Which backend to prioritize is a build-time decision. |
| Text, born-digital | Direct text extraction, no model | Free, and higher fidelity than re-OCRing text that's already machine-readable. |
| Text, scanned | Self-hosted PaddleOCR (primary) + AIN-7B (hard cases, e.g. Arabic) | Classic OCR engines are free per-call and CPU/GPU-capable; AIN-7B covers cases PaddleOCR struggles with. |
| Text, OCR fallback | Amazon Bedrock — **Claude Haiku 4.5** | Reverts to the original pick: on GCP, Claude required a Marketplace purchase plus a manually-approved quota increase, which stalled on a fresh-account fraud-prevention hold. On Bedrock, Claude is a first-party-supported model family with self-service model access (no purchase step) — the friction that forced the Gemini substitution on GCP doesn't exist here. Used only when self-hosted options fail, keeping call spend minimal. |
| Table structure | Self-hosted, pluggable across TableFormer, TATR, RapidTable, PP-StructureV3 | No managed equivalent; lightweight, self-hosting is free. TSR always runs; OCR is paired in only when the table is scanned. Which backend to prioritize is a build-time decision. |
| Figures/charts/stamps | Amazon Bedrock — **Claude Haiku 4.5** | Same rationale as above; satisfies the privacy constraint (Bedrock inference stays inside AWS's boundary within the account) and avoids self-hosting a VLM for a task with low call volume. |

**Explicitly excluded**: Qwen2.5-VL-32B (32B params — needs an expensive multi-GPU instance class, disproportionate to PoC scale), Qwen2.5-VL-7B (managed alternatives at similar cost make self-hosting a 7B VLM unnecessary for this scope), **Gemini family** (Google-first-party, not available on Bedrock — was the GCP-era pick, now excluded by the platform move itself, not a quality judgment), and **Amazon Nova (Premier/Lite)** — viable on Bedrock and no longer excluded on access grounds (the old "AWS-only, not available on GCP" exclusion was a GCP-side artifact), but not selected over Claude for this pass; revisit if Claude's Bedrock cost/quota profile turns out worse in practice than expected.

### 4.4 AIN-7B quantization
AIN-7B is int8-quantized (~7GB) rather than run at fp16 (~14GB), even though a 16GB T4 could technically fit it at fp16. This is deliberate: the local dev GPU (RTX 3500 Ada, 12GB) cannot fit the fp16 version at all, and maintaining two precision configs (fp16 in the cloud, int8 locally) risks silent behavior differences between dev and prod. Running int8 everywhere keeps dev/prod parity and leaves comfortable VRAM headroom for future stages to also use GPU acceleration.

### 4.5 Single EC2 GPU instance, manually managed
One T4-attached instance (`g4dn.xlarge`) hosts all self-hosted containers, built on AWS's Deep Learning AMI (GPU variant — NVIDIA drivers + Docker preinstalled). Auto-stop/lifecycle automation is **explicitly deferred** — the instance is started and stopped by hand around demo/dev sessions. This was a conscious tradeoff to avoid building autoscaling infrastructure (e.g., SageMaker endpoints or ECS-on-GPU with scale-to-zero) for a workload that's demo-only; see Section 7 for the cost risk this carries and the guardrail in place (AWS Budgets alerts).

### 4.6 Born-digital vs. scanned detection: page-level, not per-region
The pipeline conceptually wants per-region granularity (a page could have both born-digital text and a scanned stamp). For the PoC, this is simplified to a page-level check to reduce implementation surface. This is a known accuracy tradeoff, not a permanent design choice — flagged as an open item in the implementation plan.

## 5. Operational Watch-Outs

- **VRAM ceiling differs between dev and prod**: local Ada card (12GB) < target T4 (16GB, `g4dn.xlarge`). Anything validated locally with quantization should stay quantized in prod unless deliberately re-tested at higher precision.
- **NAT Gateway is a cost trap** — avoided entirely. The instance sits with a public IP and a security group locked to a known IP range, rather than a private-subnet-plus-NAT-Gateway setup.
- **Forgetting to stop the EC2 GPU instance is the single largest cost risk** in this architecture — a `g4dn.xlarge` left running unattended for even a couple of days could blow past the self-imposed $50 cap. Mitigated today only by an AWS Budgets alert (Section 7 notes this isn't sufficient on its own).
- **Bedrock model access still needs to be requested per model** (self-service, typically fast/automatic for Anthropic/Amazon first-party models — unlike the GCP Marketplace-purchase-plus-manual-quota-approval flow that stalled Claude access there) — request Claude Haiku 4.5 access early in Phase 0 regardless, since it's a blocking dependency for Task 0.4. **EC2 GPU (`g4dn`) quota/limit still needs to be checked/requested** for a fresh account — new AWS accounts often start with a low or zero on-demand vCPU limit for GPU instance families; sequenced early in the implementation plan for that reason, mirroring the same friction hit on both prior cloud attempts (original AWS signup, then GCP's T4 quota).
- **Container portability**: both dev (local, x86_64/Ubuntu-in-WSL2) and prod (EC2, x86_64) share the same CPU architecture, so no cross-arch builds are needed — but CUDA compute-capability coverage (Ada = sm_89, T4 = sm_75) should be verified in each framework's prebuilt wheels before assuming a locally-built image will run correctly on the T4, same caveat as before.

## 6. Security Practices Applied

Despite being a PoC, these are treated as non-negotiable because they're cheap and prevent avoidable exposure:

- **No public S3 access** on either bucket (Block Public Access enabled account-wide); default AWS-managed encryption at rest (SSE-S3).
- **Least-privilege IAM** per component (submit Lambda, status Lambda, EC2 instance role) — each scoped to only the specific S3 buckets, DynamoDB table, SQS queue, and (for the EC2 instance role) Bedrock invoke access, not broad managed policies like `AdministratorAccess`.
- **No hardcoded secrets** — all cross-service auth is via IAM roles, not access keys embedded in containers or code.
- **MFA** enabled on the AWS root account and any IAM users; day-to-day work uses scoped IAM roles, not broad root/long-lived-key credentials, wherever automation is involved.
- **Security group locked to a known IP/CIDR** — no `0.0.0.0/0` ingress on the EC2 instance.
- **Bedrock-only / self-hosted-only model usage** — satisfies the hard privacy constraint that document content never reaches a public vendor API; Bedrock inference stays inside AWS's boundary within your account.
- **Config validation before deploy** — CloudFormation templates validated (`cfn-lint`, plus `cfn-guard`/`cfn-nag` where available) before every deploy, catching public-access and over-broad IAM misconfigurations before they reach the account.

## 7. Security Practices Intentionally Deferred (PoC Tradeoffs)

These are gaps a production system would need to close. Each is called out with why it's acceptable *right now* and what changes the calculus later.

| Gap | Why it's acceptable for this PoC | What would change it |
|---|---|---|
| **No private VPC subnet / NAT Gateway** — EC2 instance has a public IP with an IP-locked security group instead of full network isolation | NAT Gateway carries an ongoing hourly + data-processing cost — disproportionate to a $200 credit and a single trusted operator's IP. The security group still blocks all unsolicited inbound traffic. | Once multiple people need access, or the instance handles anything beyond a demo, move to a private subnet with VPC endpoints (S3, DynamoDB, Bedrock) instead of a public IP. |
| **No AWS WAF / API-level rate limiting on API Gateway** | Handful of known internal users, not an internet-facing product. Abuse risk is low and the cost of WAF isn't justified yet. | Add AWS WAF and API Gateway usage plans/throttling before opening access beyond a small trusted group. |
| **No formal user authentication (Cognito/SSO) on the API** — access is via AWS IAM credentials for company-account users within the account, not a separate identity layer | Every caller is already an IAM-authenticated company employee within a controlled account; adding a separate auth layer for a handful of trusted internal users is premature. | Before wider internal rollout, add an Amazon Cognito user pool or an SSO-backed Lambda authorizer so access isn't tied to raw IAM membership. |
| **No automated EC2 shutdown** — lifecycle is fully manual | Explicitly deferred per project decision to avoid building autoscaling infrastructure before the pipeline itself is validated. | This is the highest-priority item to fix before unattended/production use — see the cost risk in Section 5. At minimum, add a scheduled idle-timeout auto-stop (EventBridge Scheduler + Lambda); longer-term, migrate to a scale-to-zero GPU serving option (e.g. SageMaker async inference). |
| **No CloudTrail review / Security Hub / GuardDuty** | Single-operator PoC account; the operator is the only actor, so detection tooling has low marginal value right now. | Enable before any additional users or shared credentials are introduced — this is cheap to turn on and should happen early in any path to production, even before other hardening. |
| **No S3 object versioning / DynamoDB point-in-time recovery** — no backup/DR strategy | PoC data is disposable/re-derivable (source docs can be re-uploaded, results re-generated); losing PoC state is an inconvenience, not an incident. | Required once real, non-reproducible company documents or results need to persist reliably. |
| **Single AZ, no redundancy** — one EC2 GPU instance, no multi-AZ failover for any component | A demo doesn't need to survive an AZ outage; the async queue already tolerates the compute being down (jobs simply wait). | Not a PoC concern; revisit only if this becomes a relied-upon internal service rather than a demo. |
| **No automated OS/container patching** | Short-lived PoC instance, rebuilt from a fresh AMI + freshly-pushed images each time it's needed rather than kept running long-term. | Needed once the instance (or its replacement) runs continuously rather than being started fresh per session. |

## 8. Path Beyond PoC (if this graduates)

In rough priority order: (1) automate EC2 lifecycle (auto-stop at minimum, scale-to-zero GPU serving longer-term), (2) enable CloudTrail / Security Hub / GuardDuty, (3) move to a private subnet with restricted egress (VPC endpoints), (4) add a real identity layer (Cognito/SSO) in front of the API, (5) add AWS WAF/throttling, (6) enable backup/versioning on persistent data stores, (7) move born-digital/scanned detection to per-region granularity. None of these block the PoC itself — they're the gap list between "works for a demo" and "safe to leave running unattended for real users."
