import { createWriteStream } from "node:fs";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import { NextResponse } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";
import { spawnBatch } from "@/lib/spawn-batch";
import type { SaveResult } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Video = { token: string; size: number; download: string };

// Where downloaded videos land: a gitignored `recordings/` folder inside the
// project (process.cwd() is the project root under `next start`). Mirrors each
// node's dataset tree under <saveRoot>/<nodeId>/. Override with SMARTROOM_SAVE_DIR.
function saveRoot() {
  return process.env.SMARTROOM_SAVE_DIR || path.join(process.cwd(), "recordings");
}

// Download every recording from one node into <root>/<nodeId>/<relative path>,
// skipping files already present at the same size (idempotent re-runs).
async function saveNode(
  node: (typeof NODES)[number],
  root: string
): Promise<SaveResult> {
  const result: SaveResult = {
    id: node.id,
    name: node.name,
    downloaded: 0,
    skipped: 0,
    failed: 0,
    bytes: 0,
  };
  try {
    const res = await fetch(`${baseUrl(node)}/recordings`, {
      signal: AbortSignal.timeout(8000),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`listing HTTP ${res.status}`);
    const { videos } = (await res.json()) as { videos: Video[] };

    for (const v of videos) {
      // Source token: day_X/rec_Y/streams/file.mp4  or  day_X/rec_Y/metadata.json.
      // Lay both cameras under one shared rec, with each node inside streams/:
      //   <root>/day_X/rec_Y/streams/<node>/<file>
      // (the Pis use identical day/rec names, so same folder == same session).
      const raw = v.token.replace(/^data\//, "");
      const [day, rec, ...rest] = raw.split("/");
      const tail = rest.filter((s) => s !== "streams"); // drop the source 'streams' segment
      const dest = path.join(root, day, rec, "streams", node.id, ...tail);
      try {
        const st = await stat(dest);
        if (st.size === v.size) {
          result.skipped++;
          continue;
        }
      } catch {
        // not present yet
      }
      try {
        await mkdir(path.dirname(dest), { recursive: true });
        const dl = await fetch(`${baseUrl(node)}${v.download}`, {
          signal: AbortSignal.timeout(120000),
          cache: "no-store",
        });
        if (!dl.ok || !dl.body) throw new Error(`download HTTP ${dl.status}`);
        const webStream = dl.body as unknown as Parameters<typeof Readable.fromWeb>[0];
        await pipeline(Readable.fromWeb(webStream), createWriteStream(dest));
        result.downloaded++;
        result.bytes += v.size;
      } catch {
        result.failed++;
      }
    }
  } catch (e) {
    result.error = e instanceof Error ? e.message : "unreachable";
  }
  return result;
}

// After both nodes' files are on disk, write one combined metadata.json in each
// recording's streams/ dir — the folder that holds cam1/ and cam2/ — merging every
// node's own per-camera metadata.json so a session's cameras are described together
// in one place. Regenerated on every Save All (idempotent). Each stream's path is
// re-rooted to the combined dir (e.g. "streams/camera_main.mp4" -> "cam1/camera_main.mp4").
async function writeCombinedMetadata(root: string) {
  const listDirs = async (dir: string) => {
    try {
      return (await readdir(dir, { withFileTypes: true }))
        .filter((e) => e.isDirectory())
        .map((e) => e.name);
    } catch {
      return [];
    }
  };
  for (const day of await listDirs(root)) {
    for (const rec of await listDirs(path.join(root, day))) {
      const streamsDir = path.join(root, day, rec, "streams");
      const cameras: Record<string, unknown> = {};
      let recordingId = rec;
      let space: unknown;
      let schemaVersion: unknown;
      for (const node of NODES) {
        let raw: Record<string, unknown>;
        try {
          raw = JSON.parse(await readFile(path.join(streamsDir, node.id, "metadata.json"), "utf8"));
        } catch {
          continue; // this node has no metadata for this rec — skip it
        }
        recordingId = (raw.recording_id as string) ?? recordingId;
        space = raw.space ?? space;
        schemaVersion = raw.schema_version ?? schemaVersion;
        const reroot = (p: unknown) =>
          typeof p === "string" ? `${node.id}/${path.basename(p)}` : p;
        const streams: Record<string, unknown> = {};
        for (const [name, s] of Object.entries((raw.streams as Record<string, unknown>) ?? {})) {
          const sv = s as Record<string, unknown>;
          streams[name] = { ...sv, path: reroot(sv.path), timestamps_path: reroot(sv.timestamps_path) };
        }
        cameras[node.id] = {
          node: raw.node,
          start_time: raw.start_time,
          end_time: raw.end_time,
          duration_seconds: raw.duration_seconds,
          streams,
        };
      }
      if (Object.keys(cameras).length === 0) continue;
      const combined = { recording_id: recordingId, space, schema_version: schemaVersion, cameras };
      try {
        await writeFile(path.join(streamsDir, "metadata.json"), JSON.stringify(combined, null, 2));
      } catch {
        // best-effort — a failed combined write shouldn't fail the save
      }
    }
  }
}

export async function POST() {
  const root = saveRoot();
  const nodes = await Promise.all(NODES.map((n) => saveNode(n, root)));
  await writeCombinedMetadata(root);
  // Lens-corrected copies for calibrated recordings (streams/<cam>/undistorted/);
  // idempotent + flocked, so firing after every save is safe. Analysis prefers
  // these copies when present. Same venv python as detection (needs cv2).
  try {
    const python =
      process.env.SMARTROOM_DETECT_PYTHON ||
      path.join(process.env.HOME || "", "Code", "yolo-bench", ".venv", "bin", "python");
    spawnBatch(python, [path.join(process.cwd(), "detect", "undistort.py")], {
      cwd: process.cwd(),
      logName: ".undistort.log",
    });
  } catch {
    // undistortion is best-effort; the save itself succeeded
  }
  return NextResponse.json(
    { saveRoot: root, nodes },
    { headers: { "cache-control": "no-store" } }
  );
}
