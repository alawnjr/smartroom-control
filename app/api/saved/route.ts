import { readFileSync } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { readDetections, readUndistorted, readValidation } from "@/lib/detections";
import { savedRoot } from "@/lib/recordings";
import type { SavedVideo } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VIDEO_EXT = new Set([".mp4", ".mov", ".mkv", ".webm", ".avi"]);

// List every saved video under recordings/, parsed into node/day/rec, newest
// first. Returns an empty list (not an error) if nothing's been saved yet.
export async function GET() {
  const root = savedRoot();
  let rels: string[] = [];
  try {
    const dirents = await readdir(root, { recursive: true, withFileTypes: true });
    rels = dirents
      .filter(
        (d) =>
          d.isFile() &&
          VIDEO_EXT.has(path.extname(d.name).toLowerCase()) &&
          !d.name.includes(".annotated.") && // outputs, not source clips
          // raw 16-bit depth data (lossless FFV1, camera_*_depth.mkv) — not
          // playable video; beamed + stored for analysis but not a clip card
          !/_depth\.\w+$/.test(d.name) &&
          !d.parentPath.split(path.sep).includes("undistorted") // lens-corrected copies, not extra clips
      )
      .map((d) => path.relative(root, path.join(d.parentPath, d.name)));
  } catch {
    rels = [];
  }

  const videos: SavedVideo[] = [];
  // Per-clip wall-clock start from the cam dir's metadata.json (streams are
  // keyed by the clip's stem). Cached per dir — several clips share one file.
  type StreamMeta = { start_time?: string; hw_clock_offset_ms?: number };
  const metaCache = new Map<string, { start_time?: string; streams?: Record<string, StreamMeta> } | null>();
  const streamInfoFor = (abs: string): { startMs?: number; hwOffsetMs?: number } => {
    const dir = path.dirname(abs);
    if (!metaCache.has(dir)) {
      try {
        metaCache.set(dir, JSON.parse(readFileSync(path.join(dir, "metadata.json"), "utf8")));
      } catch {
        metaCache.set(dir, null);
      }
    }
    const meta = metaCache.get(dir);
    const stem = path.basename(abs, path.extname(abs));
    const entry = meta?.streams?.[stem];
    const iso = entry?.start_time ?? meta?.start_time;
    const ms = iso ? Date.parse(iso) : NaN;
    return {
      startMs: Number.isFinite(ms) ? ms : undefined,
      hwOffsetMs: typeof entry?.hw_clock_offset_ms === "number" ? entry.hw_clock_offset_ms : undefined,
    };
  };

  for (const rel of rels) {
    // Layout: <day>/<rec>/streams/<node>/<file> — node lives inside streams/.
    const parts = rel.split(path.sep);
    const si = parts.indexOf("streams");
    const abs = path.join(root, rel);
    let size = 0;
    let mtime = 0;
    try {
      const st = await stat(abs);
      size = st.size;
      mtime = st.mtimeMs;
    } catch {
      // ignore unreadable file
    }
    const detections = await readDetections(abs);
    const validation = await readValidation(abs);
    const undistorted = readUndistorted(abs);
    videos.push({
      validation,
      undistortedRelPath: undistorted?.rel,
      undistortedVersion: undistorted?.version,
      node: (si >= 0 ? parts[si + 1] : "") ?? "",
      day: parts[0] ?? "",
      rec: parts[1] ?? "",
      file: parts[parts.length - 1],
      relPath: rel,
      size,
      mtime,
      ...streamInfoFor(abs),
      detections,
    });
  }
  videos.sort((a, b) => b.mtime - a.mtime);

  return NextResponse.json({ root, videos }, { headers: { "cache-control": "no-store" } });
}
