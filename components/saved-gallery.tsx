"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";

import { OccupancySparkline } from "@/components/occupancy-sparkline";
import type { DetectionSummary, SavedListing, SavedVideo } from "@/lib/types";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose"];
const MODEL_LABEL: Record<string, string> = {
  yolo26n: "nano",
  yolo26s: "small",
  yolo26m: "medium",
  yolo26l: "large",
  "yolo26n-pose": "pose",
};

function mb(b: number) {
  return `${(b / 1e6).toFixed(b >= 1e7 ? 0 : 1)} MB`;
}

function fileUrl(relPath: string) {
  return `/api/saved/file?path=${encodeURIComponent(relPath)}`;
}

function OccupancyBadge({ d }: { d?: DetectionSummary }) {
  if (!d || d.status === "none") {
    return <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-500">not analyzed</span>;
  }
  if (d.status === "analyzing") {
    return (
      <span className="flex items-center gap-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-400">
        <span className="size-1.5 animate-pulse rounded-full bg-amber-400" /> analyzing…
      </span>
    );
  }
  if (d.status === "error") {
    return (
      <span className="rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-400" title={d.error}>
        analysis failed
      </span>
    );
  }
  return (
    <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400">
      👤 max {d.maxPersons} · avg {d.avgPersons}
    </span>
  );
}

function ClipCard({ v, model }: { v: SavedVideo; model: string | null }) {
  const d = model ? v.detections?.[model] : undefined;
  const hasAnnotated = Boolean(d?.hasAnnotated && d.annotatedRelPath);
  const [annotated, setAnnotated] = useState(false);
  const showAnnotated = annotated && hasAnnotated;
  const src = showAnnotated ? fileUrl(d!.annotatedRelPath!) : fileUrl(v.relPath);

  const qc = useQueryClient();
  const reanalyze = useMutation({
    mutationFn: () =>
      fetch("/api/detect", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ relPath: v.relPath, force: true }),
      }).catch(() => {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }),
  });

  return (
    <div className="flex flex-col gap-1 rounded-lg border border-neutral-800 bg-black/30 p-2">
      <video key={src} controls preload="none" className="aspect-video w-full rounded bg-black" src={src} />
      {d?.status === "done" && d.timeline && <OccupancySparkline timeline={d.timeline} max={d.maxPersons ?? 0} />}
      <div className="flex items-center justify-between gap-2">
        <OccupancyBadge d={d} />
        <div className="flex items-center gap-1">
          {hasAnnotated && (
            <button
              onClick={() => setAnnotated((a) => !a)}
              className="rounded border border-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300 hover:bg-neutral-800"
            >
              {showAnnotated ? "raw" : "overlay"}
            </button>
          )}
          <button
            onClick={() => reanalyze.mutate()}
            disabled={reanalyze.isPending}
            title="Re-run detection on this clip (all models)"
            className="flex items-center rounded border border-neutral-700 p-1 text-neutral-400 hover:bg-neutral-800 disabled:opacity-50"
          >
            <RefreshCw className={`size-3 ${reanalyze.isPending ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between gap-2 text-xs text-neutral-400">
        <span className="truncate" title={`${v.day}/${v.rec}/${v.file}`}>{v.rec || v.file}</span>
        <a className="shrink-0 text-emerald-400 hover:underline" href={fileUrl(v.relPath)} download={`${v.node}_${v.rec}_${v.file}`}>
          download
        </a>
      </div>
      <span className="text-[10px] text-neutral-600">{v.day} · {mb(v.size)}</span>
    </div>
  );
}

export function SavedGallery() {
  const { data } = useQuery({
    queryKey: ["saved"],
    queryFn: async (): Promise<SavedListing> => {
      const res = await fetch("/api/saved", { cache: "no-store" });
      return res.json();
    },
    refetchOnWindowFocus: false,
    refetchInterval: (q) =>
      q.state.data?.videos?.some((v) =>
        Object.values(v.detections ?? {}).some((d) => d.status === "analyzing")
      )
        ? 3000
        : false,
  });

  const videos = data?.videos ?? [];

  // models present across all clips, in nano→small→medium order
  const available = MODEL_ORDER.filter((m) => videos.some((v) => v.detections?.[m]));
  const [sel, setSel] = useState<string | null>(null);
  // default to the largest available *detection* model (not pose)
  const detection = available.filter((m) => !m.includes("pose"));
  const fallback = detection[detection.length - 1] ?? available[available.length - 1] ?? null;
  const model = sel && available.includes(sel) ? sel : fallback;

  const byNode = new Map<string, SavedVideo[]>();
  for (const v of videos) {
    const arr = byNode.get(v.node);
    if (arr) arr.push(v);
    else byNode.set(v.node, [v]);
  }
  const nodes = [...byNode.keys()].sort();

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-neutral-300">Saved Recordings</h2>
        {available.length > 0 && (
          <div className="flex items-center gap-1 text-xs">
            <span className="text-neutral-500">model</span>
            <div className="flex overflow-hidden rounded-md border border-neutral-700">
              {available.map((m) => (
                <button
                  key={m}
                  onClick={() => setSel(m)}
                  className={`px-2 py-0.5 ${
                    m === model ? "bg-emerald-600 text-white" : "text-neutral-300 hover:bg-neutral-800"
                  }`}
                >
                  {MODEL_LABEL[m] ?? m}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {videos.length === 0 ? (
        <p className="text-xs text-neutral-500">No saved recordings yet — hit “Save All to Laptop”.</p>
      ) : (
        <div className="flex flex-col gap-5">
          {nodes.map((node) => (
            <div key={node} className="flex flex-col gap-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
                {node} <span className="font-normal lowercase text-neutral-600">({byNode.get(node)!.length})</span>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {byNode.get(node)!.map((v) => (
                  <ClipCard key={v.relPath} v={v} model={model} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
