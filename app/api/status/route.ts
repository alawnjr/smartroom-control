import { NextResponse } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";
import type { NodeStatus, PiStatus } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Fan out to every node's /record/status. A dead/slow node maps to
// {online:false} rather than failing the whole request, and a short per-node
// timeout keeps one unreachable node from stalling the ~1s client poll.
export async function GET() {
  const nodes: NodeStatus[] = await Promise.all(
    NODES.map(async (n): Promise<NodeStatus> => {
      try {
        const res = await fetch(`${baseUrl(n)}/record/status`, {
          signal: AbortSignal.timeout(2000),
          cache: "no-store",
        });
        if (!res.ok) {
          return { ...n, online: false, status: null, error: `HTTP ${res.status}` };
        }
        const status = (await res.json()) as PiStatus;
        return { ...n, online: true, status };
      } catch (e) {
        return {
          ...n,
          online: false,
          status: null,
          error: e instanceof Error ? e.message : "unreachable",
        };
      }
    })
  );

  return NextResponse.json({ nodes }, { headers: { "cache-control": "no-store" } });
}
