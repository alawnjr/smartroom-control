"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { SaveResponse } from "@/lib/types";

function mb(bytes: number) {
  return (bytes / 1e6).toFixed(bytes >= 1e7 ? 0 : 1);
}

export function SaveBar() {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: async (): Promise<SaveResponse> => {
      const res = await fetch("/api/save-all", { method: "POST" });
      return res.json();
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }), // refresh the gallery
  });
  const data = save.data;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="outline"
          size="lg"
          disabled={save.isPending}
          onClick={() => save.mutate()}
        >
          <Download />
          {save.isPending ? "Saving…" : "Save All to Laptop"}
        </Button>
        {data && <span className="text-xs text-neutral-500">→ {data.saveRoot}</span>}
        {save.isError && <span className="text-xs text-red-400">request failed</span>}
      </div>

      {data && (
        <ul className="flex flex-col gap-0.5 text-xs">
          {data.nodes.map((n) => (
            <li key={n.id} className={n.error ? "text-red-400" : "text-neutral-400"}>
              {n.name}:{" "}
              {n.error
                ? `error — ${n.error}`
                : `${n.downloaded} saved, ${n.skipped} already had, ${n.failed} failed (${mb(n.bytes)} MB)`}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
