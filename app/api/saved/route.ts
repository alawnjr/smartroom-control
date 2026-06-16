import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

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
      .filter((d) => d.isFile() && VIDEO_EXT.has(path.extname(d.name).toLowerCase()))
      .map((d) => path.relative(root, path.join(d.parentPath, d.name)));
  } catch {
    rels = [];
  }

  const videos: SavedVideo[] = [];
  for (const rel of rels) {
    const parts = rel.split(path.sep);
    let size = 0;
    let mtime = 0;
    try {
      const st = await stat(path.join(root, rel));
      size = st.size;
      mtime = st.mtimeMs;
    } catch {
      // ignore unreadable file
    }
    videos.push({
      node: parts[0] ?? "",
      day: parts[1] ?? "",
      rec: parts[2] ?? "",
      file: parts[parts.length - 1],
      relPath: rel,
      size,
      mtime,
    });
  }
  videos.sort((a, b) => b.mtime - a.mtime);

  return NextResponse.json({ root, videos }, { headers: { "cache-control": "no-store" } });
}
