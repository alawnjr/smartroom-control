import { execFile } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import { NextRequest, NextResponse } from "next/server";

import { readDetections, readValidation } from "@/lib/detections";
import { savedRoot } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const execFileP = promisify(execFile);

// Read-only machine API for other devices on the LAN: inference results and
// compressed frames from the saved recordings. See /api/v1 for a self-
// describing index. CORS is open — everything here is already LAN-visible.
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, OPTIONS",
  "cache-control": "no-store",
};

function json(body: unknown, status = 200) {
  return NextResponse.json(body, { status, headers: CORS });
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS });
}

// Path segments come from clients — keep them boring (no slashes/dots-dots).
const SAFE = /^[\w.-]+$/;
const safe = (s: string | undefined) => Boolean(s && SAFE.test(s) && !s.includes(".."));

// Sidecar kinds an inference response merges (suffix -> response key).
const SIDECARS: Record<string, string> = {
  detections: "detections",
  actions: "actions",
  persons: "persons",
  centroids: "centroids",
  keypoints: "keypoints",
};

function camDir(day: string, rec: string, cam: string) {
  return path.join(savedRoot(), day, rec, "streams", cam);
}

function listDirs(dir: string): string[] {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort();
  } catch {
    return [];
  }
}

async function readJsonIf(p: string) {
  try {
    return JSON.parse(await readFile(p, "utf8"));
  } catch {
    return undefined;
  }
}

// GET /api/v1/recordings — every session, newest first, with per-camera
// analysis/validation summaries and the URLs for the deeper endpoints.
async function listRecordings(origin: string) {
  const root = savedRoot();
  const sessions = [];
  for (const day of listDirs(root)) {
    for (const rec of listDirs(path.join(root, day))) {
      const streamsDir = path.join(root, day, rec, "streams");
      const cams: Record<string, unknown> = {};
      let mtime = 0;
      for (const cam of listDirs(streamsDir)) {
        const mp4 = path.join(streamsDir, cam, "camera_main.mp4");
        let st;
        try {
          st = statSync(mp4);
        } catch {
          continue;
        }
        mtime = Math.max(mtime, st.mtimeMs);
        const detections = await readDetections(mp4);
        const validation = await readValidation(mp4);
        const meta = await readJsonIf(path.join(streamsDir, cam, "metadata.json"));
        const base = `${origin}/api/v1/recordings/${day}/${rec}/${cam}`;
        cams[cam] = {
          node: meta?.node,
          startTime: meta?.start_time,
          durationSec: meta?.duration_seconds,
          calibrated: Boolean(meta?.streams?.camera_main?.calibration),
          models: Object.fromEntries(
            Object.entries(detections).map(([m, d]) => [m, d.status])
          ),
          validation: validation
            ? { status: validation.status, passed: validation.passed, failed: validation.failed }
            : null,
          urls: {
            inference: `${base}/inference/{model}`,
            frame: `${base}/frame?t={seconds}&w={width}&q={1-100}&video={raw|undistorted|annotated.<model>}`,
            video: `${base}/video?variant={raw|undistorted|annotated.<model>}`,
          },
        };
      }
      if (Object.keys(cams).length) sessions.push({ day, rec, mtime, cameras: cams });
    }
  }
  sessions.sort((a, b) => b.mtime - a.mtime);
  return json({ recordings: sessions });
}

// GET .../{day}/{rec}/{cam}/inference/{model} — every sidecar for one model,
// merged into a single response (detections summary, action timelines,
// per-person segments, centroids, raw keypoints — whichever exist).
async function inference(day: string, rec: string, cam: string, model: string) {
  const dir = camDir(day, rec, cam);
  const out: Record<string, unknown> = { day, rec, camera: cam, model };
  let any = false;
  for (const [suffix, key] of Object.entries(SIDECARS)) {
    const data = await readJsonIf(path.join(dir, `camera_main.${suffix}.${model}.json`));
    if (data !== undefined) {
      out[key] = data;
      any = true;
    }
  }
  if (!any) return json({ error: `no inference for model '${model}' on this clip` }, 404);
  const meta = await readJsonIf(path.join(dir, "metadata.json"));
  out.calibration = meta?.streams?.camera_main?.calibration ?? null;
  out.extrinsics = meta?.streams?.camera_main?.extrinsics ?? null;
  // Metric room coordinate frame (AprilTag-relative floor positions) — present
  // when the clip was analyzed with extrinsics + a known tag height. Track
  // centroid entries then carry room:[x_mm,z_mm] + src ("ankles"|"bbox").
  /* eslint-disable @typescript-eslint/no-explicit-any */
  out.roomFrame = (out.centroids as any)?.roomFrame ?? null;
  /* eslint-enable @typescript-eslint/no-explicit-any */

  // Action models: also join everything per person into a top-level `tracks`
  // array with an explicit trackId, so consumers don't have to correlate three
  // separately-keyed blobs. (The raw sidecar sections above stay as-is.)
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const timelines = (out.actions as any)?.tracks ?? {};
  const persons = (out.persons as any)?.persons ?? {};
  const cents = (out.centroids as any)?.persons ?? {};
  const ids = [...new Set([...Object.keys(timelines), ...Object.keys(persons), ...Object.keys(cents)])];
  if (ids.length) {
    ids.sort((a, b) => Number(a) - Number(b));
    out.tracks = ids.map((id) => ({
      trackId: id,
      dominantAction: (out.detections as any)?.trackActions?.[id] ?? null,
      segments: persons[id]?.segments ?? [],       // merged action ranges {action,start,end,conf}
      jumps: persons[id]?.jumps ?? [],             // geometric jump events, seconds
      timeline: timelines[id] ?? [],               // per-window {t,action,conf,kept,top}
      centroids: cents[id] ?? [],                  // per-frame {t,x,y} px (+ room:[x,z] mm, src when located)
    }));
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */
  return json(out);
}

// Which video file a variant refers to; annotated variants are per-model.
function videoPath(dir: string, variant: string): string | null {
  if (variant === "raw") return path.join(dir, "camera_main.mp4");
  if (variant === "undistorted") return path.join(dir, "undistorted", "camera_main.mp4");
  const m = variant.match(/^annotated\.([\w-]+)$/);
  if (m) return path.join(dir, `camera_main.annotated.${m[1]}.mp4`);
  return null;
}

// GET .../{day}/{rec}/{cam}/frame?t=&w=&q=&video= — one JPEG frame at time t,
// scaled to width w (default 640) at JPEG quality q (default 80). ffmpeg does
// the seek + compress; a LAN consumer polling frames stays cheap.
async function frame(dir: string, req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const t = Math.max(0, Number(sp.get("t") ?? 0) || 0);
  const w = Math.min(1920, Math.max(64, Number(sp.get("w") ?? 640) || 640));
  const q = Math.min(100, Math.max(1, Number(sp.get("q") ?? 80) || 80));
  const variant = sp.get("video") ?? "raw";
  const src = videoPath(dir, variant);
  if (!src) return json({ error: `unknown video variant '${variant}'` }, 400);
  try {
    statSync(src);
  } catch {
    return json({ error: `no '${variant}' video for this clip` }, 404);
  }
  // JPEG quality 1-100 -> ffmpeg qscale 31 (worst) .. 2 (best).
  const qscale = Math.round(31 - (q / 100) * 29);
  try {
    const { stdout } = await execFileP(
      "ffmpeg",
      ["-v", "error", "-ss", String(t), "-i", src, "-frames:v", "1",
       "-vf", `scale=${w}:-2`, "-q:v", String(qscale), "-f", "image2", "pipe:1"],
      { encoding: "buffer", maxBuffer: 16 * 1024 * 1024, timeout: 20000 }
    );
    if (!stdout.length) return json({ error: "no frame at that time (past end of clip?)" }, 416);
    return new NextResponse(new Uint8Array(stdout), {
      // Recordings are immutable once saved, so extracted frames can cache hard
      // (the dashboard uses these as video posters — no-store would re-run
      // ffmpeg for every card on every visit).
      headers: {
        ...CORS,
        "cache-control": "public, max-age=86400",
        "content-type": "image/jpeg",
        "content-length": String(stdout.length),
      },
    });
  } catch {
    return json({ error: "frame extraction failed" }, 500);
  }
}

// GET .../{day}/{rec}/{cam}/video?variant= — redirect to the (range-capable)
// file server so consumers can stream/download whole videos.
function video(day: string, rec: string, cam: string, req: NextRequest) {
  const variant = req.nextUrl.searchParams.get("variant") ?? "raw";
  const dir = camDir(day, rec, cam);
  const abs = videoPath(dir, variant);
  if (!abs) return json({ error: `unknown video variant '${variant}'` }, 400);
  try {
    statSync(abs);
  } catch {
    return json({ error: `no '${variant}' video for this clip` }, 404);
  }
  const rel = path.relative(savedRoot(), abs);
  // Manual 307 — NextResponse.redirect drops Location when given custom headers.
  return new NextResponse(null, {
    status: 307,
    headers: { ...CORS, location: `/api/saved/file?path=${encodeURIComponent(rel)}` },
  });
}

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ seg?: string[] }> }
) {
  const { seg = [] } = await ctx.params;
  if (!seg.every((s) => safe(s))) return json({ error: "bad path" }, 400);

  if (seg.length === 0) return listRecordings(req.nextUrl.origin);

  const [day, rec, cam, action, model] = seg;
  if (seg.length === 5 && action === "inference") return inference(day, rec, cam, model);
  if (seg.length === 4 && action === "frame") return frame(camDir(day, rec, cam), req);
  if (seg.length === 4 && action === "video") return video(day, rec, cam, req);
  return json({ error: "unknown endpoint; see /api/v1" }, 404);
}
