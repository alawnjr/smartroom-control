import { createWriteStream } from "node:fs";
import { mkdir, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import { NextResponse } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";
import type { SaveResult } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Video = { token: string; size: number; download: string };

// Where downloaded videos land on this laptop. Mirrors each node's dataset tree
// under <saveRoot>/<nodeId>/. Override with SMARTROOM_SAVE_DIR.
function saveRoot() {
  return (
    process.env.SMARTROOM_SAVE_DIR ||
    path.join(os.homedir(), "Videos", "Smartroom Recordings")
  );
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
      const rel = v.token.replace(/^data\//, ""); // day_X/rec_Y/streams/file.mp4
      const dest = path.join(root, node.id, rel);
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

export async function POST() {
  const root = saveRoot();
  const nodes = await Promise.all(NODES.map((n) => saveNode(n, root)));
  return NextResponse.json(
    { saveRoot: root, nodes },
    { headers: { "cache-control": "no-store" } }
  );
}
