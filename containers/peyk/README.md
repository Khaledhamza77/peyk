# peyk

Merged worker image (docs-personal/new_containerization_strategy.md) — one `--stage` flag
picks which role runs: `layout`, `tsr`, `ocr`, `dcr`, or `vlm`. Each stage's code lives
unmodified under `stages/<stage>/`, copied in from its original single-purpose image
(`peyk-layout`, `peyk-tsr`, `peyk-simple-ocr`, `peyk-dcr`, `peyk-vlm`).

This README currently documents only the `vlm` stage's credential/model details (carried over
from `peyk-vlm`'s own README) — the other stages don't need any local setup beyond the image
build itself.

## vlm stage

Thin client calling managed cloud LLM APIs (AWS Bedrock, GCP Vertex AI) for four roles. No
local GPU, no model weights: every `--model` is a remote API call.

### Models (`--model`)

See `stages/vlm/backends/registry.py`'s `MODEL_REGISTRY` — adding a new model is one dict entry
there, not a new class. **Model keys are plain model names — no `bedrock-`/`vertex-` prefix.**
Which provider a model belongs to is looked up in the registry, never guessed from the key's
spelling; run `docker run --rm peyk:dev --stage vlm --list-models` for the live
`<key>\t<provider>` list (this is exactly what `peyk-orchestrator`'s `config.py` queries to
validate a model name and decide which credentials to mount):

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

See `stages/vlm/backends/registry.py` for exact model IDs. Every Anthropic model except Haiku
4.5 needs its own one-time Bedrock agreement accepted before it's invokable; Sonnet 5/Fable
5/Opus 4.7/4.8 are gated behind an AWS-Sales tier that self-service CLI/console steps can't
clear.

### Roles (`--role`)

- `ocr` — transcribe a scanned text-region crop.
- `figure` — describe a figure/chart/stamp crop.
- `table` — recognize a whole table crop as HTML (structure + text in one call).
- `fullpage` — transcribe a whole rendered page image to Markdown.

### Local dev credentials (mounted/env, not fetched from Secrets Manager)

This container reads credentials from whatever's mounted/passed into it — nothing is fetched
from a secrets manager at container start.

- **Bedrock** (any Bedrock-provider model — check `--list-models`): **preferred — a Bedrock API key**, no code change
  needed. `bedrock-runtime`'s service model supports both SigV4 and bearer-token auth
  (`signingName: "bedrock"`, confirmed via its `service-2.json`); botocore auto-derives and
  prefers the env var `AWS_BEARER_TOKEN_BEDROCK` when set, over the normal credential chain.
  Generate one via the Bedrock console (API keys page), put it in a local untracked `.env`
  file (`containers/peyk/.env` — already covered by the repo's `.gitignore`, never commit
  it), and pass `--env-file` to `docker run`. The API key is scoped to just Bedrock, so this
  container never needs anything else. The old `~/.aws` mount + `AWS_PROFILE` approach still
  works as a fallback if you don't have an API key yet — botocore only uses the bearer token
  when `AWS_BEARER_TOKEN_BEDROCK` is set.
- **Vertex** (any `gemini-*` or `deepseek-ocr` model — check `--list-models`): bind-mount a local
  GCP service-account JSON key, set `GOOGLE_APPLICATION_CREDENTIALS` to its in-container path.
  Override `GCP_PROJECT`/`GCP_LOCATION` only if not using the project defaults
  (`peyk-501209`/`europe-west1`) baked into `stages/vlm/backends/vertex_gemini.py`/`vertex_maas.py`.

### Example invocations

```bash
docker build -t peyk:dev containers/peyk

# List every model the vlm stage supports, with its real provider
docker run --rm peyk:dev --stage vlm --list-models

# Bedrock, via API key (preferred) — containers/peyk/.env contains one line:
# AWS_BEARER_TOKEN_BEDROCK=<key>
docker run --rm \
  --env-file containers/peyk/.env \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk:dev --stage vlm --model claude-haiku --role ocr --input /input --output /output

# Bedrock, via mounted IAM profile (fallback, if no API key set up yet)
docker run --rm \
  -v ~/.aws:/root/.aws:ro -e AWS_PROFILE=peyk \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk:dev --stage vlm --model claude-haiku --role ocr --input /input --output /output

# Vertex (Gemini or DeepSeek-OCR — same credential mount for both)
docker run --rm \
  -v /path/to/gcp-key.json:/secrets/gcp-key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json \
  -v "$(pwd)/test_in":/input -v "$(pwd)/test_out":/output \
  peyk:dev --stage vlm --model gemini-2-5-flash --role figure --input /input --output /output
```
