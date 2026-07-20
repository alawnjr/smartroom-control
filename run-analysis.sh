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
#   SMARTROOM_GPUS         GPUs to shard clips across    (default: auto = nvidia-smi count)
# Positional args (override the env, for backward compat with the old runner):
#   $1 models   $2 variants   $3 force-flag (e.g. --force)
#
# Multi-GPU: when >1 GPU is available, each stage shards its clips round-robin
# across the GPUs and runs one worker per GPU in parallel (each pinned with
# CUDA_VISIBLE_DEVICES and its own lock suffix). Clips are independent (a
# recording's cameras are paired only within its own dir), so this is safe;
# stages still run in order (localize consumes detect's pose sidecars).
set -uo pipefail

WORK_DIR="${SMARTROOM_REMOTE_DIR:-/root/smartroom-control}"
cd "$WORK_DIR"
export PATH="$HOME/.local/bin:$PATH"
export SMARTROOM_SAVE_DIR="${SMARTROOM_SAVE_DIR:-$WORK_DIR/recordings}"

MODELS="${1:-${SMARTROOM_YOLO_MODELS:-yolo26m,yolo26n-pose}}"
VARIANTS="${2:-${SMARTROOM_ACTION_VARIANTS:-hmdb}}"
FORCE_FLAG="${3:-}"
FF=(); [ -n "$FORCE_FLAG" ] && FF=(--force)

GPUS="${SMARTROOM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$GPUS" -ge 1 ] 2>/dev/null || GPUS=1

# All RGB clips (recordings-relative), the unit of work every stage shards over.
mapfile -t CLIPS < <(cd "$SMARTROOM_SAVE_DIR" && \
  find . \( -name camera_main.mp4 -o -name camera_d455_color.mp4 -o -name camera_d435_color.mp4 \) \
  | grep -v '/undistorted/' | sed 's|^\./||' | sort)

# run_stage <label> <python> <script> <pathstyle: append|nargs> [extra args...]
# Shards CLIPS across $GPUS workers; each worker gets CUDA_VISIBLE_DEVICES + a
# lock suffix and its slice via --path. Falls back to a single self-discovering
# run on 1 GPU or when the clip list is empty (preserves original behaviour).
run_stage() {
  local label="$1" py="$2" script="$3" style="$4"; shift 4
  local extra=("$@")
  echo "[$(date)] === $label ${FORCE_FLAG:+(force)} · ${GPUS} GPU(s) ==="
  if [ "$GPUS" -le 1 ] || [ "${#CLIPS[@]}" -eq 0 ]; then
    "$py" "$script" "${extra[@]}" "${FF[@]}"
    return $?
  fi
  local pids=() gpu
  for ((gpu=0; gpu<GPUS; gpu++)); do
    local shard=() j
    for ((j=gpu; j<${#CLIPS[@]}; j+=GPUS)); do shard+=("${CLIPS[j]}"); done
    [ "${#shard[@]}" -eq 0 ] && continue
    local patharg=() c
    if [ "$style" = "append" ]; then
      for c in "${shard[@]}"; do patharg+=(--path "$c"); done
    else
      patharg=(--path "${shard[@]}")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" SMARTROOM_LOCK_SUFFIX=".g$gpu" \
      "$py" "$script" "${extra[@]}" "${FF[@]}" "${patharg[@]}" &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  return $rc
}

rc=0
echo "[$(date)] === analyzing $SMARTROOM_SAVE_DIR (${#CLIPS[@]} clips) ==="
export SMARTROOM_YOLO_MODELS="$MODELS"
run_stage "object detection + pose: $MODELS" detect/.venv-detect/bin/python detect/detect.py append || rc=$?
run_stage "spatial localization (pose + depth -> room positions)" detect/.venv-detect/bin/python detect/localize.py nargs || rc=$?
run_stage "action recognition: $VARIANTS" .venv-action/bin/python detect/action.py append --variant "$VARIANTS" || rc=$?
echo "[$(date)] === finished, exit=$rc ==="
echo "$rc" > "$WORK_DIR/analyze.done"
exit "$rc"
