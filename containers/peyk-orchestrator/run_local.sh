#!/usr/bin/env bash
# Runs peyk-orchestrator with the mounts it actually needs, so it behaves the same way
# locally and on EC2:
#   - the host docker socket, so it can dispatch sibling stage containers via
#     docker-outside-of-docker (see stages.py)
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
#     during the run. All of this is still local-disk-only, never S3, safe to wipe between jobs
#     — same as before, just split across two mount types.
#     Sibling stage containers reach all of it via `--volumes-from` this container's fixed name
#     (see stages.py's ORCHESTRATOR_CONTAINER_NAME) rather than a host-path bind mount of their
#     own — the orchestrator itself runs containerized, so a sibling it dispatches via the host
#     docker socket can't resolve a bind-mount source expressed as *this* container's own path
#     (e.g. "/hotstorage/...") against the host filesystem; --volumes-from sidesteps that
#     entirely by reusing this container's already-resolved mounts instead of re-deriving a
#     host path — and this works identically whether a given mount is a bind mount or a named
#     volume, so mixing the two here doesn't complicate that fix at all.
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
CONFIG_FILE="${PEYK_CONFIG:-$SCRIPT_DIR/config/example.yaml}"
IMAGE="${PEYK_ORCHESTRATOR_IMAGE:-peyk-orchestrator:dev}"

# peyk-vlm's cloud credentials — resolved to host-absolute paths (win_pwd, same as HOTSTORAGE
# above) and passed into the orchestrator container as env vars, since pipeline.py's
# _vlm_credential_docker_args needs a real host path for the *inner* docker run (see that
# function's docstring for why --volumes-from doesn't work for these). Empty string, not a
# missing/unset var, if the file doesn't exist yet (fresh checkout without credentials set
# up) — pipeline.py raises a clear error at dispatch time only if a config actually selects a
# peyk-vlm backend needing the one that's missing, not on every run regardless of config.
PEYK_VLM_DIR="$SCRIPT_DIR/../peyk-vlm"
if [ -f "$PEYK_VLM_DIR/.env" ]; then
    PEYK_VLM_ENV_FILE="$(win_pwd "$PEYK_VLM_DIR")/.env"
else
    PEYK_VLM_ENV_FILE=""
fi
if [ -f "$PEYK_VLM_DIR/gcp-key.json" ]; then
    PEYK_VLM_GCP_KEY_FILE="$(win_pwd "$PEYK_VLM_DIR")/gcp-key.json"
else
    PEYK_VLM_GCP_KEY_FILE=""
fi
# Must match stages.py's ORCHESTRATOR_CONTAINER_NAME.
CONTAINER_NAME="peyk-orchestrator-run"
# Docker creates this automatically on first use if it doesn't exist — no separate
# provisioning step needed, same as the peyk-paddlex-cache/peyk-vllm-paddleocr-cache volumes.
WORKDIR_VOLUME="peyk-hotstorage-workdir"

mkdir -p "$HOTSTORAGE/input" "$HOTSTORAGE/output"
# Re-resolved now that it's guaranteed to exist (HOTSTORAGE may be relative and/or not have
# existed before the mkdir above) — win_pwd needs to actually cd into it.
HOTSTORAGE="$(win_pwd "$HOTSTORAGE")"

# Force-remove any container left over from a previous run under this same fixed name (e.g.
# a prior run killed rather than left to exit cleanly) — same reasoning as stages.py's own
# per-stage cleanup. Errors ignored: the common case is there's nothing to remove.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

set +e
docker run --rm --name "$CONTAINER_NAME" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$SCRIPT_DIR/config:/app/config:ro" \
  -v "$HOTSTORAGE/input:/hotstorage/input" \
  -v "$HOTSTORAGE/output:/hotstorage/output" \
  -v "$WORKDIR_VOLUME:/hotstorage/workdir" \
  -e PEYK_VLM_ENV_FILE="$PEYK_VLM_ENV_FILE" \
  -e PEYK_VLM_GCP_KEY_FILE="$PEYK_VLM_GCP_KEY_FILE" \
  "$IMAGE" \
  --config "/app/config/$(basename "$CONFIG_FILE")" \
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
