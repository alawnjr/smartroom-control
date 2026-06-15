import { NextResponse } from "next/server";

import { NODES, baseUrl } from "@/lib/nodes";
import type { RecordResult } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Cancel any running recording on every node. A node's 409 ("nothing running")
// is treated as benign success.
export async function POST() {
  const results: RecordResult[] = await Promise.all(
    NODES.map(async (n): Promise<RecordResult> => {
      try {
        const res = await fetch(`${baseUrl(n)}/record/cancel`, {
          method: "POST",
          signal: AbortSignal.timeout(5000),
          cache: "no-store",
        });
        let message = "";
        try {
          message = ((await res.json()) as { message?: string })?.message ?? "";
        } catch {
          // non-JSON body
        }
        const ok = res.ok || res.status === 409; // 409 = nothing running
        return {
          id: n.id,
          name: n.name,
          ok,
          httpStatus: res.status,
          message:
            message ||
            (res.status === 409 ? "nothing running" : ok ? "cancelled" : `HTTP ${res.status}`),
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
