"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Download, Film, Loader2, ScanEye, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { analyzingCount, pingSavedSoon, useSaved } from "@/lib/use-saved";
import type { SaveResponse } from "@/lib/types";

function mb(bytes: number) {
  return (bytes / 1e6).toFixed(bytes >= 1e7 ? 0 : 1);
}

function triggerDetect(body?: { force?: boolean }) {
  // fire-and-forget; the gallery's analyzing-poll surfaces progress
  return fetch("/api/detect", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).catch(() => {});
}

export function SaveBar() {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: async (): Promise<SaveResponse> => {
      const res = await fetch("/api/save-all", { method: "POST" });
      return res.json();
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["saved"] }); // refresh the gallery
      triggerDetect(); // analyze freshly pulled clips
      pingSavedSoon(qc); // pick up the analyzing state once it starts
    },
  });
  const analyze = useMutation({
    mutationFn: () => triggerDetect({ force: true }), // re-run all clips
    onSuccess: () => pingSavedSoon(qc),
  });
  const actions = useMutation({
    mutationFn: () =>
      fetch("/api/action", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ force: true }),
      }).catch(() => {}),
    onSuccess: () => pingSavedSoon(qc),
  });
  const cancel = useMutation({
    mutationFn: () => fetch("/api/detect/cancel", { method: "POST" }).catch(() => {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }),
  });
  const saved = useSaved();
  const analyzing = analyzingCount(saved.data); // clips currently being analyzed
  const busy = analyze.isPending || analyzing > 0;
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
        <Button
          variant="outline"
          size="lg"
          disabled={busy}
          onClick={() => analyze.mutate()}
          title="Re-run person-detection over all saved recordings"
        >
          {busy ? <Loader2 className="animate-spin" /> : <ScanEye />}
          {analyzing > 0
            ? `Analyzing ${analyzing}…`
            : analyze.isPending
              ? "Starting…"
              : "Re-analyze"}
        </Button>
        <Button
          variant="outline"
          size="lg"
          disabled={busy || actions.isPending}
          onClick={() => actions.mutate()}
          title="Per-person action recognition (pose tracking + Kinetics video model; slower)"
        >
          <Film />
          {actions.isPending ? "Starting…" : "Actions"}
        </Button>
        {analyzing > 0 && (
          <Button
            variant="destructive"
            size="lg"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate()}
            title="Stop the running analysis"
          >
            <X />
            {cancel.isPending ? "Cancelling…" : "Cancel"}
          </Button>
        )}
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
