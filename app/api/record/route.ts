import { NextRequest, NextResponse } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";
import type { RecordResult } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Start a recording on every node at once. Fired in one tick to minimize
// start-skew (the clap-at-t0 marker is still the fine-sync mechanism). A node's
// 409 ("already recording") is treated as benign success.
export async function POST(req: NextRequest) {
  let duration = 30;
  try {
    const body = await req.json();
    duration = Math.max(1, Math.min(3600, Math.round(Number(body?.duration) || 30)));
  } catch {
    // no/invalid body -> default 30s
  }
  const form = new URLSearchParams({ duration: String(duration) }).toString();

  const results: RecordResult[] = await Promise.all(
    NODES.map(async (n): Promise<RecordResult> => {
      try {
        const res = await fetch(`${baseUrl(n)}/record`, {
          method: "POST",
          headers: { "content-type": "application/x-www-form-urlencoded" },
          body: form,
          signal: AbortSignal.timeout(5000),
          cache: "no-store",
        });
        let message = "";
        try {
          message = ((await res.json()) as { message?: string })?.message ?? "";
        } catch {
          // non-JSON body
        }
        const ok = res.ok || res.status === 409; // 409 = already recording
        return {
          id: n.id,
          name: n.name,
          ok,
          httpStatus: res.status,
          message:
            message ||
            (res.status === 409 ? "already recording" : ok ? "started" : `HTTP ${res.status}`),
        };
      } catch (e) {
        return {
          id: n.id,
          name: n.name,
          ok: false,
          httpStatus: null,
          message: e instanceof Error ? e.message : "unreachable",
        };
      }
    })
  );

  return NextResponse.json(
    { results, allOk: results.every((r) => r.ok) },
    { headers: { "cache-control": "no-store" } }
  );
}
