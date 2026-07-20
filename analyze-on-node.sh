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
# ${VAR-default} (no colon): SMARTROOM_GW= (explicitly empty) disables the
# gateway hop for directly-reachable nodes; unset still gets the default.
GW="${SMARTROOM_GW-alawnjr@grid.cosmos-lab.org}"
NODE="${SMARTROOM_NODE:-root@srv1}"
LOCAL_REC="${SMARTROOM_LOCAL_REC:-$HOME/Code/smartroom-control/recordings}"
REMOTE_DIR="${SMARTROOM_REMOTE_DIR:-/root/smartroom-control}"
REMOTE_REC="$REMOTE_DIR/recordings"
# Where the analysis reads/writes recordings ON THE NODE. Defaults to the repo's
# own recordings/ (old COSMOS flow: push here, analyze in place, pull back). On
# the Rutgers quad server this is the data volume, separate from the code dir:
#   SMARTROOM_SAVE_DIR=/mnt/data4/intern26/recordings
SAVE_DIR="${SMARTROOM_SAVE_DIR:-$REMOTE_REC}"
YOLO_MODELS="${SMARTROOM_YOLO_MODELS:-yolo26n,yolo26s,yolo26m,yolo26l,yolo26n-pose}"
ACTION_VARIANTS="${SMARTROOM_ACTION_VARIANTS:-hmdb}"
FORCE="${FORCE:-0}"

# SSH/rsync hop the node via the gateway (ProxyJump) unless SMARTROOM_GW is
# set EMPTY — some nodes (e.g. srv2-lg2.bed) are directly reachable.
SSHOPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
RSYNC_RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
if [ -n "$GW" ]; then
  SSHOPTS=(-J "$GW" "${SSHOPTS[@]}")
  RSYNC_RSH="ssh -J $GW -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
fi

node_ssh() { ssh "${SSHOPTS[@]}" "$NODE" "$@"; }
log()      { printf '\n\033[1;36m== %s ==\033[0m\n' "$*" >&2; }
die()      { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- bootstrap ----
bootstrap() {
  log "Bootstrapping $NODE:$REMOTE_DIR (ffmpeg, uv, detection venv, action venv) — idempotent"
  node_ssh 'bash -s' -- "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$1"

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
  echo ">> building detect/.venv-detect (torch + ultralytics, a few min)"
  rm -rf detect/.venv-detect
  uv venv --python 3.12 detect/.venv-detect
  # Pin torch to a CUDA build the box's driver actually supports BEFORE
  # ultralytics pulls the default PyPI wheel (currently cu13, which needs a very
  # new driver). cu118 covers Volta..Ada incl. the Quadro RTX 6000 (sm_75);
  # CPU wheel when there's no working GPU. Mirrors setup-action-env.sh.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo ">> GPU detected — installing cu118 torch into detect venv"
    uv pip install --python detect/.venv-detect torch torchvision --index-url https://download.pytorch.org/whl/cu118
  else
    echo ">> no GPU — installing CPU torch into detect venv"
    uv pip install --python detect/.venv-detect torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
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

# run-analysis.sh is a committed, env-driven script (not generated here) so the
# server's smartroom-analyze.service can invoke the exact same runner. Push it up
# fresh on every analyze so pass changes reach already-bootstrapped nodes.
write_runner() {
  local script_dir; script_dir="$(cd "$(dirname "$0")" && pwd)"
  [ -f "$script_dir/run-analysis.sh" ] || die "run-analysis.sh not found next to analyze-on-node.sh"
  rsync -a -e "$RSYNC_RSH" "$script_dir/run-analysis.sh" "$NODE:$REMOTE_DIR/run-analysis.sh"
  node_ssh "chmod +x '$REMOTE_DIR/run-analysis.sh'"
}

# --------------------------------------------------------------- push -------
push() {
  [ -d "$LOCAL_REC" ] || die "local recordings dir not found: $LOCAL_REC"
  local script_dir; script_dir="$(cd "$(dirname "$0")" && pwd)"
  log "Syncing analysis code -> srv1:$REMOTE_DIR/detect (laptop copy is canonical)"
  rsync -a -e "$RSYNC_RSH" --include='*.py' --include='*.txt' --include='*.sh' \
    --exclude='*' "$script_dir/detect/" "$NODE:$REMOTE_DIR/detect/"
  # action-classes.json (class toggles from the dashboard's Classes tab) is
  # gitignored, so it must ride along or the server run ignores the toggles.
  if [ -f "$script_dir/action-classes.json" ]; then
    rsync -a -e "$RSYNC_RSH" "$script_dir/action-classes.json" "$NODE:$REMOTE_DIR/"
  fi
  # The committed runner (also invoked by the server's systemd unit).
  rsync -a -e "$RSYNC_RSH" "$script_dir/run-analysis.sh" "$NODE:$REMOTE_DIR/run-analysis.sh"
  node_ssh "chmod +x '$REMOTE_DIR/run-analysis.sh'"
  log "Uploading recordings -> $NODE:$SAVE_DIR"
  node_ssh "mkdir -p '$SAVE_DIR'"
  rsync -ah --info=progress2 -e "$RSYNC_RSH" "$LOCAL_REC/" "$NODE:$SAVE_DIR/"
}

# -------------------------------------------------------------- analyze -----
analyze() {
  node_ssh "test -x '$REMOTE_DIR/detect/.venv-detect/bin/python'" \
    || die "node not bootstrapped — run: $0 bootstrap"
  write_runner
  local ff=""; [ "$FORCE" = "1" ] && ff="--force"
  log "Launching analysis on $NODE (detached; survives disconnect) — recordings: $SAVE_DIR"
  # Args: $1 workdir  $2 save-dir  $3 models  $4 variants  $5 force-flag
  node_ssh bash -s -- "$REMOTE_DIR" "$SAVE_DIR" "$YOLO_MODELS" "$ACTION_VARIANTS" "$ff" <<'REMOTE'
set -euo pipefail
cd "$1"
export SMARTROOM_REMOTE_DIR="$1"
export SMARTROOM_SAVE_DIR="$2"
if [ -f analyze.pid ] && kill -0 "$(cat analyze.pid)" 2>/dev/null; then
  echo "analysis already running (pid $(cat analyze.pid)); attaching."
  exit 0
fi
rm -f analyze.done analyze.log
# ${5:-}: ssh flattens the arg list, so an empty force-flag arg vanishes entirely
nohup ./run-analysis.sh "$3" "$4" "${5:-}" >analyze.log 2>&1 &
echo $! >analyze.pid
sleep 1
echo "launched pid $(cat analyze.pid); log: $1/analyze.log"
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
  log "Fetching results $NODE:$SAVE_DIR -> $LOCAL_REC (new sidecars + annotated videos)"
  rsync -ah --info=progress2 -e "$RSYNC_RSH" "$NODE:$SAVE_DIR/" "$LOCAL_REC/"
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
