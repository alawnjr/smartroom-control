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

# CUDA when the box has a working NVIDIA driver, else CPU. cu118 covers Volta
# (V100, sm_70) through Ampere (A100, sm_80) and is the newest CUDA with
# torch 2.0.x wheels — which the mmcv 2.0.x matrix pins us to.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  TORCH_IDX="https://download.pytorch.org/whl/cu118"
  MMCV_IDX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
  echo ">> GPU detected — installing CUDA (cu118) torch/mmcv"
else
  TORCH_IDX="https://download.pytorch.org/whl/cpu"
  MMCV_IDX="https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html"
  echo ">> no GPU — installing CPU torch/mmcv"
fi

uv pip install --python "$V" "torch==2.0.1" "torchvision==0.15.2" \
  --index-url "$TORCH_IDX"
uv pip install --python "$V" "numpy<2" mmengine importlib-metadata lapx ultralytics
uv pip install --python "$V" "mmcv==2.0.1" -f "$MMCV_IDX"
uv pip install --python "$V" "mmaction2==1.2.0"
# mmdet supplies the ROI head used by the SlowFast-AVA spatiotemporal detector
# (live_infer.py --action ava). 3.2.0 pins mmcv <2.2 so it keeps the 2.0.1 above.
uv pip install --python "$V" "mmdet==3.2.0"

# RTMPose — an optional alternative skeleton source for the action classifiers
# (selectable per analysis). Pure onnxruntime wheels, no mmpose/mmcv, so they don't
# touch the pinned matrix above. CPU by default; for an RTX GPU swap onnxruntime ->
# onnxruntime-gpu (NEVER install both — they clash on the CUDA provider).
uv pip install --python "$V" rtmlib onnxruntime

# The mmaction2 1.2.0 wheel is missing localizers/drn; stub it (we only use the
# skeleton recognizer, never the DRN localizer).
LOC=$("$V/bin/python" -c "import mmaction,os;print(os.path.join(os.path.dirname(mmaction.__file__),'models','localizers'))")
mkdir -p "$LOC/drn"
: > "$LOC/drn/__init__.py"
printf 'class DRN:\n    def __init__(self, *a, **k):\n        raise NotImplementedError("DRN localizer stubbed; unused")\n' > "$LOC/drn/drn.py"

# NTU-RGB+D 60 2D ST-GCN++ checkpoint (COCO-17 keypoints) — the "ntu" action variant.
CKPT="${SMARTROOM_STGCN_CKPT:-$HOME/Code/yolo-bench/stgcnpp_ntu60_2d.pth}"
if [ ! -f "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  curl -sL -o "$CKPT" "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
fi

# HMDB51 PoseC3D skeleton checkpoint (COCO-17 keypoints) — the "hmdb" action
# variant (adds walk/run; heavier 3D-CNN). Same per-track loop, so multi-person.
HMDB_CKPT="${SMARTROOM_HMDB_CKPT:-$HOME/Code/yolo-bench/posec3d_hmdb51.pth}"
if [ ! -f "$HMDB_CKPT" ]; then
  mkdir -p "$(dirname "$HMDB_CKPT")"
  curl -sL -o "$HMDB_CKPT" "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth"
fi

# SlowFast-AVA (per-person RGB spatiotemporal detection) for live_infer.py --action ava.
AVA_CKPT="${SMARTROOM_AVA_CKPT:-$HOME/Code/yolo-bench/slowfast_ava.pth}"
if [ ! -f "$AVA_CKPT" ]; then
  mkdir -p "$(dirname "$AVA_CKPT")"
  curl -sL -o "$AVA_CKPT" "https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth"
fi

# Pre-seed the RTMPose COCO-17 body model so the first analysis run needs no network
# (rtmlib lazily downloads/unzips into ~/.cache/rtmlib/hub/checkpoints on first use).
RTM_ONNX="${SMARTROOM_RTMPOSE_ONNX:-https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip}"
"$V/bin/python" -c "from rtmlib import RTMPose; RTMPose(onnx_model='$RTM_ONNX', model_input_size=(192,256), backend='onnxruntime', device='cpu'); print('rtmpose model cached')" || echo "warning: RTMPose pre-download failed (will download on first use)"

"$V/bin/python" -c "import torch,mmcv,mmaction,ultralytics,rtmlib,onnxruntime;print('action env OK: torch',torch.__version__,'mmcv',mmcv.__version__,'mmaction',mmaction.__version__,'onnxruntime',onnxruntime.__version__)"
echo "ntu checkpoint:  $CKPT"
echo "hmdb checkpoint: $HMDB_CKPT"
echo "rtmpose cache:   $HOME/.cache/rtmlib"
