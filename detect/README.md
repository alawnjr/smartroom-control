# detect/ — person-detection over saved recordings

`detect.py` runs one or more pretrained **YOLO26** models (OpenVINO, `intel:cpu`)
over every `recordings/<node>/.../streams/camera_main.mp4`, counting **people**
(COCO class 0). For each clip **and each model** it writes two siblings, so the
dashboard can toggle between models (nano / small / medium):

- `camera_main.detections.<model>.json` — occupancy stats + per-sampled-frame timeline
- `camera_main.annotated.<model>.mp4` — boxes burned in, re-encoded to H.264 (browser-playable)

Default models: `yolo26n, yolo26s, yolo26m, yolo26l`. Measured here (OpenVINO
intel:cpu, 640px): nano ~48 FPS, small ~18 FPS, medium ~7 FPS (large is slower
still) — all fine for offline batch.

It's idempotent (skips clips whose results are current), safe against concurrent
runs (a global `flock` on `recordings/.detect.lock`), and writes a
`status:"analyzing"` marker first so the dashboard shows progress.

## Run it

```bash
# Uses the shared yolo-bench venv by default
/home/alawn/Code/yolo-bench/.venv/bin/python detect/detect.py        # all unprocessed clips
... detect.py --path cam1/day_.../rec_.../streams/camera_main.mp4 --force   # one clip
... detect.py --force                                                # reprocess everything
```

## Auto-run (systemd user units)

```bash
cp deploy/smartroom-detect.{service,path,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now smartroom-detect.timer smartroom-detect.path
```

The **timer** (every 5 min) is the reliable watcher; the **path** unit is a
backstop; the dashboard also POSTs `/api/detect` right after a Save All, and the
**Re-analyze** button triggers it on demand. All funnel through the same script;
the flock makes overlapping triggers safe.

## Config (env)

| var | default | meaning |
|---|---|---|
| `SMARTROOM_DETECT_PYTHON` | `~/Code/yolo-bench/.venv/bin/python` | venv used by `/api/detect` and the units |
| `SMARTROOM_SAVE_DIR` | `<project>/recordings` | recordings root |
| `SMARTROOM_YOLO_MODELS` | `yolo26n,yolo26s,yolo26m` | model keys to run (dashboard toggles between them) |
| `SMARTROOM_YOLO_DIR` | `~/Code/yolo-bench` | dir holding `<key>_openvino_model/` |
| `SMARTROOM_DETECT_IMGSZ` | `640` | inference size |
| `SMARTROOM_DETECT_SAMPLE_FPS` | `5` | frames/sec analyzed (subsampling) |
| `SMARTROOM_DETECT_ANNOTATE` | `1` | set `0` for JSON-only (skip annotated video) |

Each model must already be exported under `SMARTROOM_YOLO_DIR`, e.g.:
```bash
for m in yolo26n yolo26s yolo26m yolo26l; do
  ~/Code/yolo-bench/.venv/bin/python -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='openvino')"
done
```
`detect.py` never exports in the hot path (a missing model dir is skipped with a warning).
