#!/usr/bin/env bash
#
# analyze-on-node.sh — offload the full smartroom analysis to the COSMOS node
# srv1, then bring the results home.
#
#   >>> RUN THIS FROM YOUR LAPTOP, not from the node. <<<
#   srv1 has no public IP and your laptop runs no sshd, so the node physically
#   cannot copy anything back to you. Your laptop, however, can reach srv1
#   through the grid gateway — so the laptop drives the whole thing:
#
#     1. bootstrap  — one-time: install ffmpeg/uv + build the detection and
#                     action Python venvs on srv1 (idempotent; skipped if present)
#     2. push       — rsync your recordings/ up to srv1
#     3. analyze    — run object detection + pose + action recognition on srv1
#                     (launched detached, so it survives an SSH drop)
#     4. wait       — follow progress until the node finishes
#     5. pull       — rsync the new sidecars + annotated videos back into
#                     your local recordings/
#
# Usage:
#   ./analyze-on-node.sh            # everything, end to end (default)
#   ./analyze-on-node.sh bootstrap  # just set up the node
#   ./analyze-on-node.sh push       # just upload recordings
#   ./analyze-on-node.sh analyze    # just kick off analysis (detached)
#   ./analyze-on-node.sh wait       # attach to a run already going on the node
#   ./analyze-on-node.sh pull       # just fetch results back
#
# Handy env overrides:
#   FORCE=1                 re-analyze clips even if results already exist
#   SMARTROOM_YOLO_MODELS   detection models   (default: n,s,m,l + n-pose)
#   SMARTROOM_ACTION_VARIANTS  action models   (default: hmdb — NTU retired)
#   SMARTROOM_LOCAL_REC     local recordings dir
#   SMARTROOM_GW / SMARTROOM_NODE   gateway / node ssh targets
#
set -euo pipefail

# ---------------------------------------------------------------- config ----
GW="${SMARTROOM_GW:-alawnjr@grid.cosmos-lab.org}"
NODE="${SMARTROOM_NODE:-root@srv1}"
LOCAL_REC="${SMARTROOM_LOCAL_REC:-$HOME/Code/smartroom-control/recordings}"
REMOTE_DIR="${SMARTROOM_REMOTE_DIR:-/root/smartroom-control}"
REMOTE_REC="$REMOTE_DIR/recordings"
YOLO_MODELS="${SMARTROOM_YOLO_MODELS:-yolo26n,yolo26s,yolo26m,yolo26l,yolo26n-pose}"
ACTION_VARIANTS="${SMARTROOM_ACTION_VARIANTS:-hmdb}"
FORCE="${FORCE:-0}"

# All SSH/rsync hops the node via the grid gateway (ProxyJump).
SSHOPTS=(-J "$GW" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
RSYNC_RSH="ssh -J $GW -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"

node_ssh() { ssh "${SSHOPTS[@]}" "$NODE" "$@"; }
log()      { printf '\n\033[1;36m== %s ==\033[0m\n' "$*" >&2; }
die()      { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- bootstrap ----
bootstrap() {
  log "Bootstrapping srv1 (ffmpeg, uv, detection venv, action venv) — idempotent"
  node_ssh 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /root/smartroom-control

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ">> installing ffmpeg"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg curl ca-certificates >/dev/null
fi

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv install failed"; exit 1; }

# --- detection venv (ultralytics / openvino / opencv) ---
if [ ! -x detect/.venv-detect/bin/python ]; then
  echo ">> building detect/.venv-detect (this pulls torch-cpu, a few min)"
  rm -rf detect/.venv-detect
  uv venv --python 3.12 detect/.venv-detect
  uv pip install --python detect/.venv-detect -r detect/requirements.txt
fi

# --- action venv (mmcv / mmaction2 + checkpoints) ---
if [ ! -x .venv-action/bin/python ]; then
  echo ">> building .venv-action via detect/setup-action-env.sh (heavy: mmcv/mmaction2 + weights)"
  rm -rf .venv-action
  bash detect/setup-action-env.sh
fi

echo "bootstrap complete"
REMOTE
}

# The detached runner the analyze step launches. (Re)written on EVERY analyze —
# not just at bootstrap — so pass changes here reach already-bootstrapped nodes.
write_runner() {
  node_ssh 'bash -s' <<'REMOTE'
set -euo pipefail
cd /root/smartroom-control
cat > run-analysis.sh <<'RUNNER'
#!/usr/bin/env bash
set -uo pipefail
cd /root/smartroom-control
export PATH="$HOME/.local/bin:$PATH"
export SMARTROOM_SAVE_DIR=/root/smartroom-control/recordings
MODELS="${1:-yolo26n,yolo26s,yolo26m,yolo26l,yolo26n-pose}"
VARIANTS="${2:-hmdb}"
FORCE_FLAG="${3:-}"
rc=0
echo "[$(date)] === object detection + pose: $MODELS ${FORCE_FLAG:+(force)} ==="
SMARTROOM_YOLO_MODELS="$MODELS" detect/.venv-detect/bin/python detect/detect.py $FORCE_FLAG || rc=$?
echo "[$(date)] === spatial localization (pose + depth -> room positions) ==="
detect/.venv-detect/bin/python detect/localize.py $FORCE_FLAG || rc=$?
echo "[$(date)] === action recognition: $VARIANTS ${FORCE_FLAG:+(force)} ==="
.venv-action/bin/python detect/action.py --variant "$VARIANTS" $FORCE_FLAG || rc=$?
echo "[$(date)] === finished, exit=$rc ==="
echo "$rc" > /root/smartroom-control/analyze.done
RUNNER
chmod +x run-analysis.sh
REMOTE
}

# --------------------------------------------------------------- push -------
push() {
  [ -d "$LOCAL_REC" ] || die "local recordings dir not found: $LOCAL_REC"
  local script_dir; script_dir="$(cd "$(dirname "$0")" && pwd)"
  log "Syncing analysis code -> srv1:$REMOTE_DIR/detect (laptop copy is canonical)"
  rsync -a -e "$RSYNC_RSH" --include='*.py' --include='*.txt' --include='*.sh' \
    --exclude='*' "$script_dir/detect/" "$NODE:$REMOTE_DIR/detect/"
  log "Uploading recordings -> srv1:$REMOTE_REC"
  node_ssh "mkdir -p '$REMOTE_REC'"
  rsync -ah --info=progress2 -e "$RSYNC_RSH" "$LOCAL_REC/" "$NODE:$REMOTE_REC/"
}

# -------------------------------------------------------------- analyze -----
analyze() {
  node_ssh "test -x '$REMOTE_DIR/detect/.venv-detect/bin/python'" \
    || die "node not bootstrapped — run: $0 bootstrap"
  write_runner
  local ff=""; [ "$FORCE" = "1" ] && ff="--force"
  log "Launching analysis on srv1 (detached; survives disconnect)"
  node_ssh bash -s -- "$YOLO_MODELS" "$ACTION_VARIANTS" "$ff" <<'REMOTE'
set -euo pipefail
cd /root/smartroom-control
if [ -f analyze.pid ] && kill -0 "$(cat analyze.pid)" 2>/dev/null; then
  echo "analysis already running (pid $(cat analyze.pid)); attaching."
  exit 0
fi
rm -f analyze.done analyze.log
# ${3:-}: ssh flattens the arg list, so an empty force-flag arg vanishes entirely
nohup ./run-analysis.sh "$1" "$2" "${3:-}" >analyze.log 2>&1 &
echo $! >analyze.pid
sleep 1
echo "launched pid $(cat analyze.pid); log: /root/smartroom-control/analyze.log"
REMOTE
}

# --------------------------------------------------------------- wait -------
wait_done() {
  log "Following analysis on srv1 (Ctrl-C is safe — the node keeps running)"
  local line=0
  while true; do
    # stream any new log lines
    local newlines
    newlines=$(node_ssh "tail -n +$((line+1)) '$REMOTE_DIR/analyze.log' 2>/dev/null" || true)
    if [ -n "$newlines" ]; then
      printf '%s\n' "$newlines"
      line=$(( line + $(printf '%s\n' "$newlines" | wc -l) ))
    fi
    if node_ssh "test -f '$REMOTE_DIR/analyze.done'"; then
      local rc; rc=$(node_ssh "cat '$REMOTE_DIR/analyze.done'")
      log "Node reports analysis finished (exit=$rc)"
      [ "$rc" = "0" ] || printf '\033[1;33m(non-zero exit — some clips/models may have failed; check analyze.log)\033[0m\n' >&2
      return 0
    fi
    sleep 20
  done
}

# --------------------------------------------------------------- pull -------
pull() {
  [ -d "$LOCAL_REC" ] || die "local recordings dir not found: $LOCAL_REC"
  log "Fetching results srv1 -> $LOCAL_REC (new sidecars + annotated videos)"
  rsync -ah --info=progress2 -e "$RSYNC_RSH" "$NODE:$REMOTE_REC/" "$LOCAL_REC/"
  log "Done. Reload the dashboard to see detections / actions / room maps."
}

# --------------------------------------------------------------- main -------
case "${1:-all}" in
  bootstrap) bootstrap ;;
  push)      push ;;
  analyze)   analyze ;;
  wait)      wait_done ;;
  pull)      pull ;;
  all)       bootstrap; push; analyze; wait_done; pull ;;
  *)         die "unknown command '$1' (use: bootstrap|push|analyze|wait|pull|all)" ;;
esac
