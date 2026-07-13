import type { NextRequest } from "next/server";

import { NODES, depthBaseUrl } from "@/lib/nodes";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Same one-origin MJPEG proxying as /api/stream/[id], but for the RealSense
// page's per-device streams: /api/depth-stream/<node>?s=<serial>&k=rgb|depth.
export async function GET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const node = NODES.find((n) => n.id === id);
  if (!node) return new Response("unknown node", { status: 404 });
  const serial = req.nextUrl.searchParams.get("s") ?? "";
  if (!serial) return new Response("missing ?s=<serial>", { status: 400 });
  const kind = req.nextUrl.searchParams.get("k") === "depth" ? "depth" : "rgb";
  try {
    const upstream = await fetch(
      `${depthBaseUrl(node)}/${kind}.mjpg?s=${encodeURIComponent(serial)}`,
      { cache: "no-store" }
    );
    if (!upstream.ok || !upstream.body) return new Response("stream unavailable", { status: 502 });
    return new Response(upstream.body, {
      status: 200,
      headers: {
        "content-type":
          upstream.headers.get("content-type") ?? "multipart/x-mixed-replace; boundary=frame",
        "cache-control": "no-store",
      },
    });
  } catch {
    return new Response("stream error", { status: 502 });
  }
}
