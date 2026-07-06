#!/usr/bin/env bash
# Build the dedicated Python 3.10 venv (.venv-action) for per-person NTU action
# recognition: mmcv/mmaction2 (won't install on the py3.14 detection venv) +
# ultralytics. Run from the project root. Needs `uv` (https://astral.sh/uv).
#
# The version matrix is exact on purpose — mmaction2 1.2.0 is the only one with
# inference_skeleton, and it needs mmcv 2.0.x, which needs torch 2.0.x.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
V=.venv-action

uv venv --python 3.10 "$V"
uv pip install --python "$V" "torch==2.0.1" "torchvision==0.15.2" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$V" "numpy<2" mmengine importlib-metadata lapx ultralytics
uv pip install --python "$V" "mmcv==2.0.1" \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html
uv pip install --python "$V" "mmaction2==1.2.0"

# The mmaction2 1.2.0 wheel is missing localizers/drn; stub it (we only use the
# skeleton recognizer, never the DRN localizer).
LOC=$("$V/bin/python" -c "import mmaction,os;print(os.path.join(os.path.dirname(mmaction.__file__),'models','localizers'))")
mkdir -p "$LOC/drn"
: > "$LOC/drn/__init__.py"
printf 'class DRN:\n    def __init__(self, *a, **k):\n        raise NotImplementedError("DRN localizer stubbed; unused")\n' > "$LOC/drn/drn.py"

# NTU-RGB+D 60 2D ST-GCN++ checkpoint (COCO-17 keypoints).
CKPT="${SMARTROOM_STGCN_CKPT:-$HOME/Code/yolo-bench/stgcnpp_ntu60_2d.pth}"
if [ ! -f "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  curl -sL -o "$CKPT" "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
fi

"$V/bin/python" -c "import torch,mmcv,mmaction,ultralytics;print('action env OK: torch',torch.__version__,'mmcv',mmcv.__version__,'mmaction',mmaction.__version__)"
echo "checkpoint: $CKPT"
