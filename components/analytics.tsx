"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LineChart, Loader2, RefreshCw, ScanEye, Video, X } from "lucide-react";

import { ClipAnalyticsDrawer } from "@/components/clip-analytics-drawer";
import { GeometricPanel } from "@/components/geometric-page";
import { OccupancyGraph } from "@/components/occupancy-graph";
import { tagClass } from "@/lib/action-colors";
import { analyzingCount, clipAnalyzing, groupSessions, pingSavedSoon, useSaved } from "@/lib/use-saved";
import type { NodeConfig, SavedVideo } from "@/lib/types";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose", "action", "action-hmdb"];
const MODEL_LABEL: Record<string, string> = {
  yolo26n: "nano", yolo26s: "small", yolo26m: "medium", yolo26l: "large",
  "yolo26n-pose": "pose", action: "actions (NTU)", "action-hmdb": "actions (HMDB)",
};
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
  const isAction = model.startsWith("action");
  const actionVariant = model === "action-hmdb" ? "hmdb" : "ntu";
  const hasOverlay = Boolean(d?.hasAnnotated && d.annotatedRelPath);
  const [overlay, setOverlay] = useState(true);
  const [drawer, setDrawer] = useState(false);
  const showOverlay = overlay && hasOverlay;
  const src = showOverlay ? fileUrl(d!.annotatedRelPath!) : fileUrl(v.relPath);
  const analyzing = clipAnalyzing(v);

  const reanalyze = useMutation({
    mutationFn: () =>
      isAction
        ? post("/api/action", { relPath: v.relPath, force: true, variant: actionVariant })
        : post("/api/detect", { relPath: v.relPath, force: true }),
    onSuccess: () => pingSavedSoon(qc),
  });

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-extrabold">
          {roomName} <span className="font-mono text-xs font-normal text-muted">· {v.rec.split("_").pop()}</span>
        </div>
        <div className="flex items-center gap-2">
          {isAction && d?.status === "done" && (
            <button
              onClick={() => setDrawer(true)}
              title="Open live graphs"
              className="flex items-center gap-1 rounded-lg border border-line px-2 py-1.5 text-[11px] font-bold text-muted hover:bg-background"
            >
              <LineChart className="size-3.5" /> Graphs
            </button>
          )}
          <button
            onClick={() => reanalyze.mutate()}
            disabled={reanalyze.isPending || analyzing}
            title="Re-run this model on this clip"
            className="rounded-lg border border-line p-1.5 text-muted hover:bg-background disabled:opacity-50"
          >
            <RefreshCw className={`size-3.5 ${reanalyze.isPending || analyzing ? "animate-spin" : ""}`} />
          </button>
        </div>
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
              <span key={t} className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${tagClass(i)}`}>{t}</span>
            ))
          )}
        </div>
      )}
      {drawer && d && (
        <ClipAnalyticsDrawer v={v} model={model} d={d} roomName={roomName} onClose={() => setDrawer(false)} />
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
  const [view, setView] = useState<"models" | "geometric">("models");

  const detectAll = useMutation({ mutationFn: () => post("/api/detect", { force: true }), onSuccess: () => pingSavedSoon(qc) });
  const actionAll = useMutation({ mutationFn: () => post("/api/action", { force: true, variant: "ntu" }), onSuccess: () => pingSavedSoon(qc) });
  const actionAllHmdb = useMutation({ mutationFn: () => post("/api/action", { force: true, variant: "hmdb" }), onSuccess: () => pingSavedSoon(qc) });
  const cancel = useMutation({ mutationFn: () => post("/api/detect/cancel"), onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }) });

  return (
    <div>
      {/* sub-view: model analysis vs. geometric (classifier-independent) events */}
      <div className="mb-4 flex overflow-hidden rounded-xl border border-line w-fit">
        {(["models", "geometric"] as const).map((vw) => (
          <button
            key={vw}
            onClick={() => setView(vw)}
            className={`px-3.5 py-1.5 text-sm font-bold ${view === vw ? "bg-foreground text-background" : "text-muted hover:bg-card"}`}
          >
            {vw === "models" ? "Models" : "Geometric"}
          </button>
        ))}
      </div>

      {view === "geometric" ? (
        <GeometricPanel nodes={config} />
      ) : (
        <>
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
            title="Per-person actions on all clips (ST-GCN++ / NTU-RGB+D 60)"
            className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
          >
            <Video className="size-4" /> Actions (NTU)
          </button>
          <button
            onClick={() => actionAllHmdb.mutate()}
            disabled={analyzing > 0 || actionAllHmdb.isPending}
            title="Per-person actions on all clips (PoseC3D / HMDB51 — adds walk/run, slower)"
            className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
          >
            <Video className="size-4" /> Actions (HMDB)
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
        <div className="flex flex-col gap-7">
          {groupSessions(videos).map((s) => (
            <div key={s.key}>
              <div className="mb-2 flex items-center gap-2 text-sm font-bold text-muted">
                <span className="font-mono">{s.label}</span>
                <span className="rounded-full bg-card px-2 py-0.5 text-xs">
                  {s.clips.length} cam{s.clips.length > 1 ? "s" : ""}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {s.clips.map((v) => (
                  <AnalysisCard key={v.relPath} v={v} model={model} roomName={nameByNode.get(v.node) ?? v.node} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
        </>
      )}
    </div>
  );
}
