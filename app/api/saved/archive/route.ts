import { spawn } from "node:child_process";
import { stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

import { safeResolve } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Stream a whole recording folder (e.g. <day>/<rec>, both cameras + sidecars +
// metadata) as a .zip. Uses bsdtar's zip writer so there's no npm/zip dependency.
// Path-traversal guarded via safeResolve.
export async function GET(req: NextRequest) {
  const rel = req.nextUrl.searchParams.get("path");
  if (!rel) return new Response("missing path", { status: 400 });
  const abs = safeResolve(rel);
  if (!abs) return new Response("bad path", { status: 400 });

  try {
    if (!(await stat(abs)).isDirectory()) return new Response("not a folder", { status: 400 });
  } catch {
    return new Response("not found", { status: 404 });
  }

  // Archive the folder nested under its own name (-C parent, then basename), so
  // the zip expands to <name>/... rather than a pile of loose files.
  const parent = path.dirname(abs);
  const name = path.basename(abs);
  const child = spawn("bsdtar", ["--format", "zip", "-cf", "-", "-C", parent, name], {
    stdio: ["ignore", "pipe", "ignore"],
  });

  const filename = rel.replace(/[\\/]+/g, "_") + ".zip";
  return new Response(Readable.toWeb(child.stdout) as unknown as ReadableStream<Uint8Array>, {
    headers: {
      "content-type": "application/zip",
      "content-disposition": `attachment; filename="${filename}"`,
      "cache-control": "no-store",
    },
  });
}
