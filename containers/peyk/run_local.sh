#!/usr/bin/env bash
# Runs peyk (the merged worker+orchestrator image) with the mounts it actually needs, so it
# behaves the same way locally and on EC2. Before docs-personal/new_containerization_strategy.md's
# step 5, this ran peyk-orchestrator as its own container, which dispatched every stage as a
# sibling container over docker-outside-of-docker (host docker socket, --volumes-from,
# per-stage container naming/cleanup, a peyk-net bootstrap race between peyk-orchestrator and
# the vLLM sidecars). All of that is gone now: peyk-orchestrator's own dispatch logic moved
# in-process (stages/orchestrator/, stage_dispatch.py) — this is the ONE container for the whole
# pipeline, so mounts/flags below apply to it directly instead of needing to be threaded through
# to nested sibling containers.
#   - --gpus all: previously only added to nested per-stage `docker run`s (stages.py's own
#     `gpu` flag) since peyk-orchestrator itself never touched a GPU directly — now that every
#     stage runs in this same process, THIS container needs GPU access itself.
#   - --network peyk-net: so the vlm/surya/paddleocr-vl stages can reach the still-separate
#     peyk-vllm-surya/peyk-vllm-paddleocr sidecars by container name. Bootstrapped here
#     (docker network create, idempotent) since there's no longer a per-stage dispatch to do it
#     from — must exist before those sidecars are started too; whichever comes up first creates
#     it.
#   - -v peyk-paddlex-cache:/root/.paddlex: PaddleX's model cache (layout/ocr backends) —
#     previously mounted into each nested per-stage container by stages.py; now just mounted
#     directly onto this one container.
#   - config/ bind-mounted OVER the image's baked-in default, so editing config/example.yaml
#     (or pointing PEYK_CONFIG at a different file) takes effect on the next run with no
#     rebuild
#   - /hotstorage/input and /hotstorage/output are host bind mounts (default host dir:
#     <repo>/hotstorage; override with PEYK_HOTSTORAGE, e.g. an EC2 instance-store path) —
#     you need to drop source PDFs in and read results out from the host, so these stay
#     Explorer/Finder-browsable. /hotstorage/workdir (every intermediate artifact: crops,
#     tsr_in/out, ocr_in/out, ...) is a Docker NAMED VOLUME instead, not a bind mount — measured
#     real per-file overhead writing many small crop files (a 376-cell table took 3.5s just for
#     PIL crop+save, ~9ms/file) through a Windows Docker Desktop bind mount's host-filesystem
#     boundary; a named volume lives inside Docker's own Linux VM and skips that boundary
#     entirely. Tradeoff: workdir/ is no longer casually browsable from Windows Explorer while
#     the run is in progress — fixed by mirroring the volume back onto $HOTSTORAGE/workdir on
#     the host after every run (see the bottom of this script), so it's still there to browse
#     once the run finishes, at the cost of one extra copy at the end rather than per-file
#     during the run. All of this is still local-disk-only, never S3, safe to wipe between jobs.
#   - vlm stage credentials (containers/peyk/.env for Bedrock, gcp-key.json for Vertex) are
#     mounted directly onto this one container now too — see below.
#
# Usage: ./run_local.sh [-- <extra docker run args>]
set -euo pipefail

# On git-bash/MSYS, bare absolute-path CLI arguments (no colon in the token) get silently
# rewritten to a Windows path by the shell before docker ever sees them (e.g. `/hotstorage/input`
# becomes `C:/.../Git/hotstorage/input`) — the exact same gotcha peyk-vllm-paddleocr/start.sh
# hit and documented. MSYS_NO_PATHCONV disables that rewriting outright; a no-op on Linux/WSL/
# macOS. Since this also suppresses MSYS's (otherwise reasonable) auto-translation of `-v
# host:container` mount arguments, the host side of every mount below is resolved to its
# Windows-native form explicitly via `pwd -W` (falls back to plain `pwd` on WSL/Linux, where
# `-W` doesn't exist) instead of relying on that auto-translation.
export MSYS_NO_PATHCONV=1

# `cd "$1" && pwd -W` on its own line (not chained with a trailing `&& pwd` fallback in the
# same expression) — `A && B || C && D` associates as `((A && B) || C) && D` in bash, so a
# one-liner "pwd -W || pwd" fallback written as part of a longer chain runs BOTH branches
# whenever the first succeeds, silently concatenating two path lines into one mangled string.
# Learned the hard way; kept as a real function specifically so nothing later touches this again.
win_pwd() {
    cd "$1" || return 1
    pwd -W 2>/dev/null || pwd
}

SCRIPT_DIR="$(win_pwd "$(dirname "${BASH_SOURCE[0]}")")"
HOTSTORAGE="${PEYK_HOTSTORAGE:-$SCRIPT_DIR/../../hotstorage}"
CONFIG_FILE="${PEYK_CONFIG:-$SCRIPT_DIR/stages/orchestrator/config/example.yaml}"
IMAGE="${PEYK_IMAGE:-peyk:dev}"

# vlm stage's cloud credentials — resolved to host-absolute paths (win_pwd, same as HOTSTORAGE
# above) and mounted/passed directly onto this one container. Empty/absent, not an error, if a
# file doesn't exist yet (fresh checkout without credentials set up) —
# pipeline.py._validate_vlm_credentials raises a clear error at dispatch time only if a config
# actually selects a vlm backend needing the one that's missing, not on every run regardless of
# config.
PEYK_VLM_DIR="$SCRIPT_DIR"
ENV_FILE_ARGS=()
if [ -f "$PEYK_VLM_DIR/.env" ]; then
    ENV_FILE_ARGS=(--env-file "$(win_pwd "$PEYK_VLM_DIR")/.env")
fi
GCP_KEY_ARGS=()
if [ -f "$PEYK_VLM_DIR/gcp-key.json" ]; then
    GCP_KEY_ARGS=(-v "$(win_pwd "$PEYK_VLM_DIR")/gcp-key.json:/secrets/gcp-key.json:ro" -e "GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json")
fi

# Must match this run's --name below.
CONTAINER_NAME="peyk-run"
# Docker creates this automatically on first use if it doesn't exist — no separate
# provisioning step needed, same as the peyk-paddlex-cache volume below.
WORKDIR_VOLUME="peyk-hotstorage-workdir"
PADDLEX_CACHE_VOLUME="peyk-paddlex-cache"
PEYK_NETWORK="peyk-net"

mkdir -p "$HOTSTORAGE/input" "$HOTSTORAGE/output"
# Re-resolved now that it's guaranteed to exist (HOTSTORAGE may be relative and/or not have
# existed before the mkdir above) — win_pwd needs to actually cd into it.
HOTSTORAGE="$(win_pwd "$HOTSTORAGE")"

# Force-remove any container left over from a previous run under this same fixed name (e.g.
# a prior run killed rather than left to exit cleanly). Errors ignored: the common case is
# there's nothing to remove.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# Idempotent: errors (already exists) ignored. Must exist before peyk-vllm-surya/
# peyk-vllm-paddleocr are started too — whichever comes up first creates it.
docker network create "$PEYK_NETWORK" >/dev/null 2>&1 || true

set +e
docker run --rm --name "$CONTAINER_NAME" \
  --gpus all \
  --network "$PEYK_NETWORK" \
  -v "$SCRIPT_DIR/stages/orchestrator/config:/app/stages/orchestrator/config:ro" \
  -v "$HOTSTORAGE/input:/hotstorage/input" \
  -v "$HOTSTORAGE/output:/hotstorage/output" \
  -v "$WORKDIR_VOLUME:/hotstorage/workdir" \
  -v "$PADDLEX_CACHE_VOLUME:/root/.paddlex" \
  "${ENV_FILE_ARGS[@]}" \
  "${GCP_KEY_ARGS[@]}" \
  "$IMAGE" \
  --config "/app/stages/orchestrator/config/$(basename "$CONFIG_FILE")" \
  --input /hotstorage/input \
  --output /hotstorage/output \
  --workdir /hotstorage/workdir \
  "$@"
RUN_EXIT=$?
set -e

# Mirrors the (fast, but not Explorer-browsable) workdir volume back onto the host so
# intermediate artifacts from THIS run — layout visualizations, TSR structure viz, OCR crops —
# are always at $HOTSTORAGE/workdir afterward, replacing whatever was there before rather than
# merging with it (a stale file left over from an earlier run wouldn't be trustworthy to leave
# lying around, same reasoning stages.py's own per-stage output-dir clearing uses). Runs even
# if the pipeline itself failed above, since that's often exactly when you want to see how far
# it got. `alpine` is pulled once (a few MB) the first time this runs.
rm -rf "$HOTSTORAGE/workdir"
mkdir -p "$HOTSTORAGE/workdir"
docker run --rm -v "$WORKDIR_VOLUME:/w:ro" -v "$HOTSTORAGE/workdir:/out" alpine cp -r /w/. /out/

exit $RUN_EXIT
