#!/usr/bin/env bash
# run-analysis.sh — the full smartroom analysis pass, in place over a recordings
# tree: object detection + pose, then spatial localization, then action
# recognition. Each stage is idempotent (skips clips whose results are already
# fresh) and writes its sidecars/annotated mp4s next to the source videos.
#
# Env-driven so the SAME script serves two callers:
#   1. analyze-on-node.sh (laptop-driven, detached over ssh — legacy COSMOS flow)
#   2. smartroom-analyze.service (server-side systemd, auto-fired when new
#      recordings land — the quad-server flow)
#
# Config (all optional; defaults preserve the original COSMOS behaviour):
#   SMARTROOM_REMOTE_DIR   repo/working dir to cd into   (default: /root/smartroom-control)
#   SMARTROOM_SAVE_DIR     recordings root to analyze    (default: $SMARTROOM_REMOTE_DIR/recordings)
#   SMARTROOM_YOLO_MODELS  detection models              (default: m + n-pose)
#   SMARTROOM_ACTION_VARIANTS  action variants           (default: hmdb)
# Positional args (override the env, for backward compat with the old runner):
#   $1 models   $2 variants   $3 force-flag (e.g. --force)
set -uo pipefail

WORK_DIR="${SMARTROOM_REMOTE_DIR:-/root/smartroom-control}"
cd "$WORK_DIR"
export PATH="$HOME/.local/bin:$PATH"
export SMARTROOM_SAVE_DIR="${SMARTROOM_SAVE_DIR:-$WORK_DIR/recordings}"

MODELS="${1:-${SMARTROOM_YOLO_MODELS:-yolo26m,yolo26n-pose}}"
VARIANTS="${2:-${SMARTROOM_ACTION_VARIANTS:-hmdb}}"
FORCE_FLAG="${3:-}"

rc=0
echo "[$(date)] === analyzing $SMARTROOM_SAVE_DIR ==="
echo "[$(date)] === object detection + pose: $MODELS ${FORCE_FLAG:+(force)} ==="
SMARTROOM_YOLO_MODELS="$MODELS" detect/.venv-detect/bin/python detect/detect.py $FORCE_FLAG || rc=$?
echo "[$(date)] === spatial localization (pose + depth -> room positions) ==="
detect/.venv-detect/bin/python detect/localize.py $FORCE_FLAG || rc=$?
echo "[$(date)] === action recognition: $VARIANTS ${FORCE_FLAG:+(force)} ==="
.venv-action/bin/python detect/action.py --variant "$VARIANTS" $FORCE_FLAG || rc=$?
echo "[$(date)] === finished, exit=$rc ==="
echo "$rc" > "$WORK_DIR/analyze.done"
exit "$rc"
