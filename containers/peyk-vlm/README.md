# peyk-vlm

Thin client calling managed cloud LLM APIs (AWS Bedrock, GCP Vertex AI) for four roles — see
`docs/implementation_plan.md` Task 1.6 and `docs/pipeline.md` for the full design/rationale.
No local GPU, no model weights: every `--model` is a remote API call.

**Wired into `peyk-orchestrator`** (Task 1.7) — `ocr`/`tsr`+`cell_ocr`/`figures`/`fullpage` in
its config can all name any model below directly. This README's examples remain useful for
standalone testing via direct `docker run`.

## Models (`--model`)

See `backends/registry.py`'s `MODEL_REGISTRY` — adding a new model is one dict entry there, not
a new class. **Model keys are plain model names — no `bedrock-`/`vertex-` prefix.** Which
provider a model belongs to is looked up in the registry, never guessed from the key's
spelling; run `docker run --rm peyk-vlm:dev --list-models` for the live `<key>\t<provider>`
list (this is exactly what `peyk-orchestrator`'s `config.py` queries to validate a model name
and decide which credentials to mount):

| `--model` | Provider | Model |
|---|---|---|
| `claude-haiku` | AWS Bedrock | Claude Haiku 4.5 |
| `claude-sonnet-4`/`4-5`/`4-6`/`5` | AWS Bedrock | Claude Sonnet 4 / 4.5 / 4.6 / 5 |
| `claude-opus-4-5`/`4-6`/`4-7`/`4-8` | AWS Bedrock | Claude Opus 4.5 / 4.6 / 4.7 / 4.8 |
| `claude-fable-5` | AWS Bedrock | Claude Fable 5 |
| `nova-lite`/`nova-pro`/`nova-2-lite` | AWS Bedrock | Amazon Nova Lite / Pro / 2 Lite |
| `pixtral-large` | AWS Bedrock | Mistral Pixtral Large |
| `kimi-k2-5` | AWS Bedrock | Moonshot AI Kimi K2.5 |
| `gemini-2-5-flash`/`2-5-flash-lite`/`2-5-pro` | GCP Vertex AI | Gemini 2.5 Flash / Flash-Lite / Pro |
| `gemini-3-flash`/`3-1-flash-lite`/`3-1-pro`/`3-5-flash` | GCP Vertex AI | Gemini 3.x family |
| `deepseek-ocr` | GCP Vertex AI (Model Garden MaaS) | DeepSeek-OCR |

See `backends/registry.py` for exact model IDs. Every Anthropic model except Haiku 4.5 needs
its own one-time Bedrock agreement accepted before it's invokable (`implementation_plan.md`
Task 1.6 has the CLI commands); Sonnet 5/Fable 5/Opus 4.7/4.8 are gated behind an AWS-Sales
tier that self-service CLI/console steps can't clear. Live-quality findings for the `ocr` role
(one Arabic financial-heading crop, not a general verdict): Sonnet 4.5/4.6, Opus 4.5/4.6
transcribed it byte-exact; Kimi K2.5 near-exact; Haiku had one character error; Nova 2 Lite had
one wrong word; Nova Lite, Nova Pro, and Pixtral Large all hallucinated unrelated text.

## Roles (`--role`)

- `ocr` — transcribe a scanned text-region crop.
- `figure` — describe a figure/chart/stamp crop.
- `table` — recognize a whole table crop as HTML (structure + text in one call).
- `fullpage` — transcribe a whole rendered page image to Markdown.

## Local dev credentials (mounted/env, not fetched from Secrets Manager)

Per `implementation_plan.md` Task 1.6's local-dev auth decision: this container reads
credentials from whatever's mounted/passed into it. The documented cross-cloud design (GCP
service-account key held in AWS Secrets Manager, fetched at container start) is Phase 3/EC2
scope — nothing to fetch-at-container-start until that IAM/Secrets Manager setup exists.

- **Bedrock** (any Bedrock-provider model — check `--list-models`): **preferred — a Bedrock API key**, no code change
  needed. `bedrock-runtime`'s service model supports both SigV4 and bearer-token auth
  (`signingName: "bedrock"`, confirmed via its `service-2.json`); botocore auto-derives and
  prefers the env var `AWS_BEARER_TOKEN_BEDROCK` when set, over the normal credential chain.
  Generate one via the Bedrock console (API keys page), put it in a local untracked `.env`
  file (`containers/peyk-vlm/.env` — already covered by the repo's `.gitignore`, never commit
  it), and pass `--env-file` to `docker run`. This is strictly better for local dev than the
  original approach (mounting the full `peyk-cicd`/`PowerUserAccess` IAM user's credential
  file) — the API key is scoped to just Bedrock, and this container never needs anything else.
  The old `~/.aws` mount + `AWS_PROFILE` approach still works as a fallback if you don't have
  an API key yet — botocore only uses the bearer token when `AWS_BEARER_TOKEN_BEDROCK` is set.
- **Vertex** (any `gemini-*` or `deepseek-ocr` model — check `--list-models`): bind-mount a local
  GCP service-account JSON key, set `GOOGLE_APPLICATION_CREDENTIALS` to its in-container path.
  Override `GCP_PROJECT`/`GCP_LOCATION` only if not using the project defaults
  (`peyk-501209`/`europe-west1`) baked into `backends/vertex_gemini.py`/`vertex_maas.py`.

## Example invocations

```bash
docker build -t peyk-vlm:dev containers/peyk-vlm

# List every model this container supports, with its real provider
docker run --rm peyk-vlm:dev --list-models

# Bedrock, via API key (preferred) — containers/peyk-vlm/.env contains one line:
# AWS_BEARER_TOKEN_BEDROCK=<key>
docker run --rm \
  --env-file containers/peyk-vlm/.env \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk-vlm:dev --model claude-haiku --role ocr --input /input --output /output

# Bedrock, via mounted IAM profile (fallback, if no API key set up yet)
docker run --rm \
  -v ~/.aws:/root/.aws:ro -e AWS_PROFILE=peyk \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk-vlm:dev --model claude-haiku --role ocr --input /input --output /output

# Vertex (Gemini or DeepSeek-OCR — same credential mount for both)
docker run --rm \
  -v /path/to/gcp-key.json:/secrets/gcp-key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk-vlm:dev --model gemini-2-5-flash --role figure --input /input --output /output
```
