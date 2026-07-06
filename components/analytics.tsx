"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, RefreshCw, ScanEye, Video, X } from "lucide-react";

import { OccupancyGraph } from "@/components/occupancy-graph";
import { analyzingCount, clipAnalyzing, pingSavedSoon, useSaved } from "@/lib/use-saved";
import type { NodeConfig, SavedVideo } from "@/lib/types";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose", "action"];
const MODEL_LABEL: Record<string, string> = {
  yolo26n: "nano", yolo26s: "small", yolo26m: "medium", yolo26l: "large",
  "yolo26n-pose": "pose", action: "actions",
};
const TAG = ["bg-amber-200 text-amber-900", "bg-sky-200 text-sky-900", "bg-violet-200 text-violet-900", "bg-emerald-200 text-emerald-900", "bg-rose-200 text-rose-900"];

function fileUrl(relPath: string) {
  return `/api/saved/file?path=${encodeURIComponent(relPath)}`;
}
function post(url: string, body?: unknown) {
  return fetch(url, {
    method: "POST",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).catch(() => {});
}

function AnalysisCard({ v, model, roomName }: { v: SavedVideo; model: string; roomName: string }) {
  const qc = useQueryClient();
  const d = v.detections?.[model];
  const isPose = model.includes("pose");
  const isAction = model === "action";
  const hasOverlay = Boolean(d?.hasAnnotated && d.annotatedRelPath);
  const [overlay, setOverlay] = useState(true);
  const showOverlay = overlay && hasOverlay;
  const src = showOverlay ? fileUrl(d!.annotatedRelPath!) : fileUrl(v.relPath);
  const analyzing = clipAnalyzing(v);

  const reanalyze = useMutation({
    mutationFn: () =>
      isAction
        ? post("/api/action", { relPath: v.relPath, force: true })
        : post("/api/detect", { relPath: v.relPath, force: true }),
    onSuccess: () => pingSavedSoon(qc),
  });

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-extrabold">
          {roomName} <span className="font-mono text-xs font-normal text-muted">· {v.rec.split("_").pop()}</span>
        </div>
        <button
          onClick={() => reanalyze.mutate()}
          disabled={reanalyze.isPending || analyzing}
          title="Re-run this model on this clip"
          className="rounded-lg border border-line p-1.5 text-muted hover:bg-background disabled:opacity-50"
        >
          <RefreshCw className={`size-3.5 ${reanalyze.isPending || analyzing ? "animate-spin" : ""}`} />
        </button>
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
        {d?.status === "done" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <video key={src} controls preload="none" className="h-full w-full object-contain" src={src} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-neutral-400">
            {d?.status === "analyzing" ? "analyzing…" : d?.status === "error" ? "analysis failed" : "not analyzed"}
          </div>
        )}
        {hasOverlay && (
          <button
            onClick={() => setOverlay((o) => !o)}
            className="absolute bottom-2 right-2 rounded-md border border-white/30 bg-black/60 px-2 py-0.5 text-[10px] font-bold text-white"
          >
            {showOverlay ? "raw" : isPose ? "skeleton" : isAction ? "labels" : "boxes"}
          </button>
        )}
      </div>

      {/* graph / tags */}
      {d?.status === "done" && !isAction && d.timeline && (
        <div className="mt-2">
          <OccupancyGraph timeline={d.timeline} max={d.maxPersons ?? 0} />
          <div className="mt-1 text-xs font-bold text-muted">
            peak {d.maxPersons} · avg {d.avgPersons} people
          </div>
        </div>
      )}
      {d?.status === "done" && isAction && (
        <div className="mt-2 flex flex-wrap gap-1">
          {(d.actions ?? []).length === 0 ? (
            <span className="text-xs text-muted">no actions detected</span>
          ) : (
            (d.actions ?? []).slice(0, 8).map((t, i) => (
              <span key={t} className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${TAG[i % TAG.length]}`}>{t}</span>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function Analytics({ nodes: config }: { nodes: NodeConfig[] }) {
  const qc = useQueryClient();
  const saved = useSaved();
  const videos = saved.data?.videos ?? [];
  const analyzing = analyzingCount(saved.data);

  const nameByNode = new Map(config.map((n) => [n.id, n.name]));

  const available = MODEL_ORDER.filter((m) => videos.some((v) => v.detections?.[m]));
  const [sel, setSel] = useState<string | null>(null);
  const model = sel && available.includes(sel) ? sel : (available[0] ?? "yolo26n");

  const detectAll = useMutation({ mutationFn: () => post("/api/detect", { force: true }), onSuccess: () => pingSavedSoon(qc) });
  const actionAll = useMutation({ mutationFn: () => post("/api/action", { force: true }), onSuccess: () => pingSavedSoon(qc) });
  const cancel = useMutation({ mutationFn: () => post("/api/detect/cancel"), onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }) });

  return (
    <div>
      {/* controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {available.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-muted">model</span>
            <div className="flex overflow-hidden rounded-xl border border-line">
              {available.map((m) => (
                <button
                  key={m}
                  onClick={() => setSel(m)}
                  className={`px-3 py-1.5 text-sm font-bold ${m === model ? "bg-emerald-500 text-white" : "bg-card text-foreground hover:bg-background"}`}
                >
                  {MODEL_LABEL[m] ?? m}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => detectAll.mutate()}
            disabled={analyzing > 0}
            className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
          >
            {analyzing > 0 ? <Loader2 className="size-4 animate-spin" /> : <ScanEye className="size-4" />}
            {analyzing > 0 ? `Analyzing ${analyzing}…` : "Re-analyze all"}
          </button>
          <button
            onClick={() => actionAll.mutate()}
            disabled={analyzing > 0 || actionAll.isPending}
            className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
          >
            <Video className="size-4" /> Actions
          </button>
          {analyzing > 0 && (
            <button
              onClick={() => cancel.mutate()}
              disabled={cancel.isPending}
              className="flex items-center gap-1.5 rounded-xl bg-rose-200 px-3 py-1.5 text-sm font-bold text-rose-800 hover:bg-rose-300"
            >
              <X className="size-4" /> Cancel
            </button>
          )}
        </div>
      </div>

      {videos.length === 0 ? (
        <p className="text-sm text-muted">No clips to analyze yet — “Beam to laptop” on the Live tab first.</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {videos.map((v) => (
            <AnalysisCard key={v.relPath} v={v} model={model} roomName={nameByNode.get(v.node) ?? v.node} />
          ))}
        </div>
      )}
    </div>
  );
}
