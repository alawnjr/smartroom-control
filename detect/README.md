# detect/ — person-detection over saved recordings

`detect.py` runs a pretrained **YOLO26 nano** model (OpenVINO, `intel:cpu`) over
every `recordings/<node>/.../streams/camera_main.mp4`, counting **people** (COCO
class 0), and writes two siblings next to each clip:

- `camera_main.detections.json` — occupancy stats + per-sampled-frame timeline
- `camera_main.annotated.mp4` — boxes burned in, re-encoded to H.264 (browser-playable)

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
| `SMARTROOM_YOLO_MODEL` | `~/Code/yolo-bench/yolo26n_openvino_model` | pre-exported OpenVINO model dir |
| `SMARTROOM_DETECT_IMGSZ` | `640` | inference size |
| `SMARTROOM_DETECT_SAMPLE_FPS` | `5` | frames/sec analyzed (subsampling) |
| `SMARTROOM_DETECT_ANNOTATE` | `1` | set `0` for JSON-only (skip annotated video) |

The OpenVINO model must already be exported (`YOLO('yolo26n.pt').export(format='openvino')`);
`detect.py` never exports in the hot path.
