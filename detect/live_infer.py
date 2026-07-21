#!/usr/bin/env python3
"""
Live inference service (server side of the live-stream feature).

The Pi cannot be reached from the quad server (one-way network), so the Pi's
`live_forward.py` PUSHES JPEG frames here over a single persistent connection,
length-prefixed:  [4-byte big-endian uint32 length][JPEG bytes] repeated.

This process runs the same YOLO26 pose model the batch pipeline uses, localizes
every person to the shared AprilTag room frame with the monocular floor-ray
(exactly `localize.py`'s `camera_main` / depth-missing path), and serves:

  POST /ingest?cam=<stream-key>   frame sink (from the Pi forwarder)
  GET  /live.mjpg                 annotated MJPEG (skeletons + foot markers)
  GET  /positions                 latest room positions JSON + roomFrame
  GET  /                          a viewer page (video + top-down room map)

Calibration is NOT sent from the Pi. Extrinsics are static, so `geom` is built
once from the newest UPLOADED recording that contains this camera (its
metadata.json already embeds calibration + extrinsics) via
`calib_utils.load_room_geometry` — the same function the batch localizer uses.

Env:
  SMARTROOM_SAVE_DIR        recordings root (to find a clip for calibration)
  SMARTROOM_DETECT_DEVICE   torch device ("0" for GPU, "cpu"); default auto
  SMARTROOM_LIVE_WEIGHTS    pose weights (default ~/Code/yolo-bench/yolo26n-pose.pt)

Usage:
  python detect/live_infer.py --cam camera_d455_color --port 8010
"""

import argparse
import json
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calib_utils import (ANKLE_JOINT_HEIGHT_MM, load_room_geometry,  # noqa: E402
                         pixel_to_floor)
from localize import Tracks, ground_point  # noqa: E402

BOUNDARY = "frame"
JPEG_QUALITY = 75
KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))

# COCO-17 skeleton edges (for drawing) + a color per limb group.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (5, 6), (11, 12), (5, 11), (6, 12),       # torso
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),  # head
]


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def find_calib_clip(cam_key: str) -> Path | None:
    """Newest uploaded <cam_key>.mp4 whose sibling metadata.json has extrinsics."""
    root = saved_root()
    if not root.exists():
        return None
    clips = sorted(root.rglob(f"{cam_key}.mp4"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for mp4 in clips:
        if "undistorted" in mp4.parts:
            continue
        md = mp4.parent / "metadata.json"
        if not md.exists():
            continue
        try:
            streams = json.loads(md.read_text()).get("streams", {})
            entry = streams.get(mp4.stem, {})
            if entry.get("calibration") and entry.get("extrinsics"):
                return mp4
        except (OSError, ValueError):
            continue
    return None


class Shared:
    """Newest-frame-wins slots shared across the ingest, inference and HTTP
    threads (mirrors realsense_depth_page.py's ViewCache pattern)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.in_jpeg = None          # latest raw JPEG bytes from the Pi
        self.in_id = 0
        self.out_jpeg = None         # latest annotated JPEG
        self.out_id = 0
        self.positions = []          # [{id,x,z,conf}]
        self.updated_ms = 0
        self.fps = 0.0

    def put_in(self, jpeg):
        with self.cond:
            self.in_jpeg = jpeg
            self.in_id += 1
            self.cond.notify_all()

    def put_out(self, jpeg, positions, fps):
        with self.cond:
            self.out_jpeg = jpeg
            self.out_id += 1
            self.positions = positions
            self.fps = fps
            self.updated_ms = int(time.time() * 1000)
            self.cond.notify_all()


def infer_loop(shared: Shared, geom: dict, weights: str, device: str, flip: bool):
    from ultralytics import YOLO
    model = YOLO(weights)
    tracks = Tracks()
    use_half = device not in ("cpu", "intel:cpu")
    last_id = 0
    ema_fps = 0.0
    print(f"[live] model loaded ({weights}) device={device} half={use_half}", flush=True)
    while True:
        with shared.cond:
            while shared.in_id == last_id or shared.in_jpeg is None:
                shared.cond.wait(timeout=5.0)
                if shared.in_jpeg is None:
                    continue
            last_id = shared.in_id
            jpeg = shared.in_jpeg
        t0 = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        try:
            res = model.predict(frame, imgsz=640, device=device,
                                half=use_half, verbose=False)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[live] predict error: {exc}", flush=True)
            continue

        persons = []
        kp = res.keypoints
        if kp is not None and kp.xyn is not None:
            xyn = kp.xyn.tolist()
            xy = kp.xy.tolist()
            conf = (kp.conf.tolist() if kp.conf is not None
                    else [[1.0] * len(p) for p in xyn])
            for i in range(len(xyn)):
                persons.append({"kpts": xyn[i], "conf": conf[i], "px": xy[i]})

        # localize each person by the floor-ray through its ankle pixel
        found = []
        for p in persons:
            ank = ground_point(p, w, h)
            if ank is None:
                continue
            hit = pixel_to_floor(ank[0], ank[1], geom, ANKLE_JOINT_HEIGHT_MM)
            if hit is None:
                continue
            found.append((hit, ank, p))
        ids = tracks.assign(t0, [f[0] for f in found])

        positions = []
        for tid, (pos, ank, p) in zip(ids, found):
            positions.append({"id": int(tid),
                              "x": round(float(pos[0]), 1),
                              "z": round(float(pos[1]), 1)})
            _draw_person(frame, p["px"], p["conf"], ank, tid)

        dt = time.time() - t0
        ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        cv2.putText(frame, f"{len(positions)} person(s)  {ema_fps:4.1f} fps",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            shared.put_out(enc.tobytes(), positions, round(ema_fps, 1))


def _draw_person(frame, px, conf, ank, tid):
    color = _track_color(tid)
    for a, b in SKELETON:
        if a < len(conf) and b < len(conf) and conf[a] > KP_CONF and conf[b] > KP_CONF:
            pa = (int(px[a][0]), int(px[a][1]))
            pb = (int(px[b][0]), int(px[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for j in range(len(conf)):
        if conf[j] > KP_CONF:
            cv2.circle(frame, (int(px[j][0]), int(px[j][1])), 3, color, -1)
    cv2.circle(frame, (int(ank[0]), int(ank[1])), 6, (0, 165, 255), 2)
    cv2.putText(frame, f"#{tid}", (int(ank[0]) + 8, int(ank[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _track_color(tid):
    rng = (37 * (tid + 1)) % 180
    hsv = np.uint8([[[rng, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def make_handler(shared: Shared, room_frame: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/ingest":
                self.send_error(404)
                return
            # length-prefixed JPEG stream over one persistent connection
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            n = 0
            try:
                while True:
                    hdr = self._readn(4)
                    if not hdr:
                        break
                    (length,) = struct.unpack(">I", hdr)
                    if length == 0 or length > 20_000_000:
                        break
                    jpeg = self._readn(length)
                    if jpeg is None:
                        break
                    shared.put_in(jpeg)
                    n += 1
            except (ConnectionError, OSError):
                pass
            print(f"[live] ingest connection closed after {n} frames", flush=True)

        def _readn(self, n):
            buf = b""
            while len(buf) < n:
                chunk = self.rfile.read(n - len(buf))
                if not chunk:
                    return None if not buf else None
                buf += chunk
            return buf

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._page()
            elif path == "/positions":
                self._positions()
            elif path == "/live.mjpg":
                self._stream()
            else:
                self.send_error(404)

        def _positions(self):
            with shared.cond:
                body = json.dumps({
                    "positions": shared.positions,
                    "updatedMs": shared.updated_ms,
                    "fps": shared.fps,
                    "roomFrame": room_frame,
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache, no-store")
            self._cors()
            self.end_headers()
            last = 0
            try:
                while True:
                    with shared.cond:
                        while shared.out_id == last or shared.out_jpeg is None:
                            shared.cond.wait(timeout=5.0)
                        last = shared.out_id
                        frame = shared.out_jpeg
                    self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (ConnectionError, OSError):
                pass

        def _page(self):
            body = PAGE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


PAGE_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>smartroom · live inference</title>
<style>
 body{margin:0;background:#0c0a09;color:#e7e5e4;font:14px system-ui;padding:16px}
 h1{font-size:16px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
 .card{background:#1c1917;border-radius:12px;padding:10px}
 img{display:block;border-radius:8px;max-width:640px;width:100%}
 canvas{background:#0a0a0a;border-radius:8px}
 .meta{font-size:12px;color:#a8a29e;margin-top:6px}
</style></head><body>
<h1>smartroom — live pose + room localization</h1>
<div class=wrap>
 <div class=card><div>Camera</div><img id=v src="/live.mjpg"><div class=meta id=fps></div></div>
 <div class=card><div>Top-down room map (mm)</div>
   <canvas id=map width=420 height=420></canvas>
   <div class=meta id=cnt></div></div>
</div>
<script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');
let room=null;
function draw(pos){
  const W=cv.width,H=cv.height,pad=30;
  ctx.clearRect(0,0,W,H);
  // room frame: X (right) horizontal, Z (out of wall) vertical (0 at wall, grows toward viewer)
  const R=4500; // mm half-extent shown
  function tx(x){return pad+(x+R)/(2*R)*(W-2*pad);}
  function tz(z){return H-pad-(z)/(R)*(H-2*pad);} // z 0..R from top wall down
  ctx.strokeStyle='#44403c';ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  ctx.fillStyle='#57534e';ctx.font='11px system-ui';
  ctx.fillText('wall / tag (z=0)',pad,pad-8);
  // camera marker
  if(room&&room.cameraPositionMm){const c=room.cameraPositionMm;
    ctx.fillStyle='#0ea5e9';ctx.beginPath();ctx.arc(tx(c[0]),tz(c[2]),5,0,7);ctx.fill();
    ctx.fillText('cam',tx(c[0])+7,tz(c[2]));}
  for(const p of pos){
    ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(tx(p.x),tz(p.z),8,0,7);ctx.fill();
    ctx.fillStyle='#0c0a09';ctx.fillText('#'+p.id,tx(p.x)-6,tz(p.z)+4);
  }
}
async function poll(){
  try{const r=await fetch('/positions');const d=await r.json();
    room=d.roomFrame;draw(d.positions||[]);
    document.getElementById('fps').textContent='inference '+(d.fps||0)+' fps';
    document.getElementById('cnt').textContent=(d.positions||[]).length+' person(s) localized';
  }catch(e){}
  setTimeout(poll,200);
}
poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cam", default="camera_d455_color",
                    help="stream key to localize (finds calibration in recordings)")
    ap.add_argument("--clip", help="explicit recording mp4 for calibration (optional)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--flip", action="store_true",
                    help="rotate incoming frames 180 (if the Pi serves unrotated)")
    args = ap.parse_args()

    weights = os.environ.get("SMARTROOM_LIVE_WEIGHTS") or str(
        Path.home() / "Code/yolo-bench/yolo26n-pose.pt")
    device = os.environ.get("SMARTROOM_DETECT_DEVICE")
    if not device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    clip = Path(args.clip) if args.clip else find_calib_clip(args.cam)
    if clip is None or not clip.exists():
        print(f"[live] FATAL: no recording with calibration for '{args.cam}' "
              f"under {saved_root()}", file=sys.stderr)
        return 2
    geom = load_room_geometry(clip, args.width, args.height, undistorted=False)
    if geom is None:
        print(f"[live] FATAL: {clip} has no room geometry (extrinsics)",
              file=sys.stderr)
        return 2
    room_frame = {
        "cameraPositionMm": [round(float(v), 1) for v in geom["cam_pos_mm"]],
        "tagId": geom.get("tag_id"),
        "tagHeightMm": geom.get("tag_height_mm"),
        "cameraId": geom.get("camera_id"),
        "calibClip": str(clip.relative_to(saved_root())),
    }
    print(f"[live] geom from {clip}  cam_pos_mm={room_frame['cameraPositionMm']}",
          flush=True)

    shared = Shared()
    threading.Thread(target=infer_loop,
                     args=(shared, geom, weights, device, args.flip),
                     daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port),
                                make_handler(shared, room_frame))
    print(f"[live] serving on :{args.port}  (POST /ingest, GET /live.mjpg, /positions, /)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
