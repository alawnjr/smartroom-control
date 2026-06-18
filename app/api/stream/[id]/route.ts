import type { NextRequest } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Proxy a node's live MJPEG stream through this app so the browser only ever
// talks to one origin (the app). This is what makes remote hosting work: over a
// Tailscale Funnel / any HTTPS tunnel the Pis aren't directly reachable and
// http streams would be blocked as mixed content — relaying them here solves
// both. No request timeout: the stream is open-ended, the browser closing the
// <img> aborts it.
export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const node = NODES.find((n) => n.id === id);
  if (!node) return new Response("unknown node", { status: 404 });
  try {
    const upstream = await fetch(`${baseUrl(node)}/stream.mjpg`, { cache: "no-store" });
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
