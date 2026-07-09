# smartroom-control LAN API (v1)

Read-only HTTP API for other devices on the lab network to consume the smart
room's **inference results** (person detection, pose, action recognition) and
**compressed video frames** from saved recordings.

- **Base URL**: `http://<laptop-ip>:4000/api/v1` — e.g. `http://10.61.1.166:4000/api/v1`.
  Use the IP, not `feruzgay.local` (that mDNS name has a conflict on this network
  and resolves unreliably). The IP is DHCP-assigned and can change after the
  laptop reconnects; re-check with `ip addr` if requests start failing.
- **No authentication** — anyone on the LAN can read. Nothing here mutates state.
- **CORS**: open (`Access-Control-Allow-Origin: *`), so browser apps can call it directly.
- **Self-describing**: `GET /api/v1` returns this endpoint map with live URLs;
  `GET /api/v1/docs` returns this document.
- Errors are JSON: `{"error": "<what went wrong>"}` with a 4xx/5xx status.

## Quickstart

```bash
BASE=http://10.61.1.166:4000/api/v1

curl $BASE/recordings                          # what exists?
curl $BASE/recordings/day_08_2026-07-07/rec_20260707_001/cam2/inference/action-hmdb
curl "$BASE/recordings/day_08_2026-07-07/rec_20260707_001/cam2/frame?t=5&w=320&q=40" -o frame.jpg
curl -L "$BASE/recordings/day_08_2026-07-07/rec_20260707_001/cam2/video?variant=raw" -o clip.mp4
```

---

## `GET /recordings` — list all recording sessions

Newest first. A *session* is one synchronized capture across the cameras
(`cam1` = smartroom1, `cam2` = smartroom2; a session may have either or both).

```jsonc
{
  "recordings": [
    {
      "day": "day_08_2026-07-07",          // folder names — use verbatim in URLs
      "rec": "rec_20260707_002",
      "mtime": 1783448292039.5,            // ms epoch of the newest file (sort key)
      "cameras": {
        "cam2": {
          "node": "smartroom2",            // hostname of the Pi that recorded it
          "startTime": "2026-07-07T14:15:56.640378-04:00",
          "durationSec": 30,
          "calibrated": false,             // camera intrinsics embedded? (see below)
          "models": {                      // analyses that exist and their status:
            "yolo26l": "done",             //   done | analyzing | error | none
            "yolo26n-pose": "done"
          },
          "validation": null,              // or {status, passed, failed} from the data validator
          "urls": { "inference": "...", "frame": "...", "video": "..." }  // templates, ready to fill
        }
      }
    }
  ]
}
```

**Models** you may see:

| key | what it is |
|---|---|
| `yolo26l` | YOLO26-large person detection (boxes, occupancy counts) |
| `yolo26n-pose` | YOLO26-nano pose (COCO-17 skeletons per sampled frame) |
| `action` | per-person action recognition — ST-GCN++ trained on NTU-RGB+D 60 |
| `action-hmdb` | per-person action recognition — PoseC3D trained on HMDB51 |

---

## `GET /recordings/{day}/{rec}/{cam}/inference/{model}` — all inference data

Merges every analysis artifact for one clip + model into one response. Keys are
present only when that artifact exists for the model:

| key | present for | contents |
|---|---|---|
| `detections` | all models | summary: status, occupancy timeline, run settings |
| `tracks` | action models | **per-person join with explicit `trackId`** — start here |
| `actions` | action models | per-track action timeline with per-window top-K |
| `persons` | action models | per-person segments + per-window keypoints |
| `centroids` | action models | per-frame body-center track (location over time) |
| `keypoints` | `yolo26n-pose` | raw normalized skeletons per sampled frame |
| `calibration` | calibrated clips | camera intrinsics (else `null`) |

### Conventions used everywhere

- **Time** `t` is **seconds from clip start** (float). Derived from the clip's
  *true* average fps, so it lines up with video playback time.
- **Coordinates** are **pixels** in the video frame, origin top-left, x right,
  y down — except the `keypoints` sidecar, which is **normalized 0–1** (multiply
  by frame width/height).
- **Skeletons** are **COCO-17** joints, in this order: nose, left-eye, right-eye,
  left-ear, right-ear, left-shoulder, right-shoulder, left-elbow, right-elbow,
  left-wrist, right-wrist, left-hip, right-hip, left-knee, right-knee,
  left-ankle, right-ankle. Each joint is `[x, y, confidence]`.
- **Track ids** are strings ("1", "2", …) assigned by the tracker; stable within
  a clip, not across clips or cameras.

### `detections` (summary)

```jsonc
{
  "status": "done",                 // done | analyzing | error
  "sourceVideo": "raw",             // "undistorted" if analysis ran on the lens-corrected copy
  "poseSource": "rtmpose",          // skeleton source: "yolo" | "rtmpose" (action models)
  "stride": 1,                      // native frames between skeleton samples
  "samplesPerClassify": 12,         // new samples between classifications
  "tracks": 2,                      // people tracked (action models)
  "trackActions": {"1": "pour"},    // dominant action per track
  "actions": ["pour", "turn"],      // distinct confident labels seen
  "jumps": 0,                       // geometric jump events (classifier-independent)
  "durationSec": 30.0,
  "timeline": [{"t": 0.0, "count": 2}, ...]   // detection models: people per sampled frame
}
```

### `tracks` (per-person join — the easiest section to consume)

One object per tracked person, everything correlated by explicit `trackId`:

```jsonc
{
  "tracks": [
    {
      "trackId": "1",                    // same id used as the key in actions/persons/centroids
      "dominantAction": "pour",          // majority vote over confident windows
      "segments": [ {"action": "pour", "start": 0.167, "end": 0.567, "conf": 0.31}, ... ],
      "jumps":    [ {"start": 3.2, "end": 3.6, "peak": 0.31} ],
      "timeline": [ {"t": 0.167, "action": "pour", "conf": 0.361, "kept": true, "top": [...]}, ... ],
      "centroids":[ {"t": 0.0, "x": 512.5, "y": 477.0}, ... ]   // per-frame body center, pixels
    }
  ]
}
```

The sections below carry the same data in its raw sidecar layout (keyed by
track id), kept for compatibility; `tracks` is the recommended entry point.

### `actions` (per-track timeline)

```jsonc
{
  "nativeFps": 29.97, "window": 48, "stride": 1,
  "tracks": {
    "1": [
      { "t": 0.167,               // seconds; label describes the motion AROUND this time
        "action": "pour",         // label, or "idle" (abstained) or "not fully in frame"
        "conf": 0.361,            // calibrated confidence of the top class
        "kept": true,             // false = low-confidence window (excluded from summaries)
        "top": [["pour",0.361],["smile",0.121], ...]   // top-K class probabilities
      }, ...
    ]
  },
  "jumps": {"1": [{"start": 3.2, "end": 3.6, "peak": 0.31}]}   // seconds; peak = rise/body-height
}
```

### `persons` (per-person view of the same data)

```jsonc
{
  "persons": {
    "1": {
      "segments": [ {"action": "pour", "start": 0.167, "end": 0.567, "conf": 0.31}, ... ],
      "windows":  [ {"t": 0.167, "action": "pour", "conf": 0.361, "kept": true,
                     "keypoints": [[530.1, 373.6, 0.905], ...]},   // 17 joints, pixels
                    ... ],
      "jumps": [ {"start": 3.2, "end": 3.6, "peak": 0.31} ]
    }
  }
}
```

### `centroids` (location tracking)

```jsonc
{ "persons": { "1": [ {"t": 0.0, "x": 512.5, "y": 477.0}, ... ] } }  // bbox center, pixels, every frame
```

### `keypoints` (raw pose stream, `yolo26n-pose` only)

```jsonc
{ "keypointFormat": "coco17_xyn",       // normalized 0-1 — multiply by frame size
  "sampleFps": 5,
  "frames": [ {"t": 0.0, "persons": [ {"kpts": [[0.41,0.52], ...], "conf": [0.98, ...]} ]} ] }
```

### `calibration` (when the camera was checkerboard-calibrated)

```jsonc
{ "camera_id": "usb-046d_0809_8633D7D7",
  "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],   // pixels
  "dist_coeffs": [k1,k2,p1,p2,k3],
  "image_size": [800, 600], "rms": 0.42, "calibrated_at": "2026-07-07T..." }
```

Note: when `detections.sourceVideo == "undistorted"`, all coordinates above are
already in **lens-corrected** pixel space (matching the `undistorted` video
variant, same camera matrix). When `"raw"`, they're in raw-frame space.

---

## `GET /recordings/{day}/{rec}/{cam}/frame` — one compressed JPEG frame

| param | default | meaning |
|---|---|---|
| `t` | `0` | time in seconds from clip start |
| `w` | `640` | output width in px (64–1920; height keeps aspect) |
| `q` | `80` | JPEG quality 1–100 |
| `video` | `raw` | `raw` \| `undistorted` \| `annotated.<model>` (e.g. `annotated.action-hmdb`) |

Returns `image/jpeg`. Typical sizes: 640px/q80 ≈ 15 KB, 320px/q40 ≈ 3 KB —
cheap enough to poll. Frames are extracted on demand (nothing pre-generated).
`416` = `t` past the end of the clip; `404` = that video variant doesn't exist
(e.g. `undistorted` for an uncalibrated recording, or `annotated.X` before
model X has run).

---

## `GET /recordings/{day}/{rec}/{cam}/video` — whole video

`?variant=raw | undistorted | annotated.<model>`. Responds `307` to a
range-capable file endpoint — use `curl -L` / any HTTP client that follows
redirects; Range requests (seeking, partial download) are supported at the
target.

---

## Recipes

**Poll the newest recording's occupancy:**

```python
import requests
BASE = "http://10.61.1.166:4000/api/v1"
newest = requests.get(f"{BASE}/recordings").json()["recordings"][0]
day, rec = newest["day"], newest["rec"]
for cam in newest["cameras"]:
    inf = requests.get(f"{BASE}/recordings/{day}/{rec}/{cam}/inference/yolo26l").json()
    tl = inf["detections"]["timeline"]
    print(cam, "peak people:", max(p["count"] for p in tl))
```

**Grab a thumbnail strip of a clip:**

```bash
for t in 0 5 10 15 20 25; do
  curl -s "$BASE/recordings/$DAY/$REC/cam2/frame?t=$t&w=320&q=60" -o "thumb_$t.jpg"
done
```

**Reconstruct a person's skeleton animation** — use
`inference/action-hmdb → persons["<id>"].windows[*].keypoints` (17×[x,y,conf]
in pixels, one entry per classified window at `windows[*].t`).
