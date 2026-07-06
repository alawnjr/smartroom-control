import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

import { safeResolve } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Serve a saved video file by its recordings-relative path, with HTTP Range
// support so the browser <video> element can seek. Path-traversal guarded.
function toBody(stream: ReturnType<typeof createReadStream>) {
  return Readable.toWeb(stream) as unknown as ReadableStream<Uint8Array>;
}

export async function GET(req: NextRequest) {
  const rel = req.nextUrl.searchParams.get("path");
  if (!rel) return new Response("missing path", { status: 400 });
  const abs = safeResolve(rel);
  if (!abs) return new Response("bad path", { status: 400 });

  let size: number;
  try {
    size = (await stat(abs)).size;
  } catch {
    return new Response("not found", { status: 404 });
  }

  const range = req.headers.get("range");
  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    let start = m && m[1] ? parseInt(m[1], 10) : 0;
    let end = m && m[2] ? parseInt(m[2], 10) : size - 1;
    if (Number.isNaN(start) || start < 0) start = 0;
    if (Number.isNaN(end) || end >= size) end = size - 1;
    if (start > end) {
      start = 0;
      end = size - 1;
    }
    return new Response(toBody(createReadStream(abs, { start, end })), {
      status: 206,
      headers: {
        "content-type": "video/mp4",
        "content-length": String(end - start + 1),
        "content-range": `bytes ${start}-${end}/${size}`,
        "accept-ranges": "bytes",
        "cache-control": "no-store",
      },
    });
  }

  return new Response(toBody(createReadStream(abs)), {
    headers: {
      "content-type": "video/mp4",
      "content-length": String(size),
      "accept-ranges": "bytes",
      "cache-control": "no-store",
    },
  });
}
