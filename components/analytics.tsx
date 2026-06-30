"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, LineChart, Loader2, Plus, RefreshCw, ScanEye, Trash2, Video, X } from "lucide-react";

import { ClipAnalyticsDrawer } from "@/components/clip-analytics-drawer";
import { GeometricPanel } from "@/components/geometric-page";
import { OccupancyGraph } from "@/components/occupancy-graph";
import { tagClass } from "@/lib/action-colors";
import { analyzingCount, clipAnalyzing, groupSessions, pingSavedSoon, useSaved, type Session } from "@/lib/use-saved";
import type { NodeConfig, SavedVideo } from "@/lib/types";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose", "action", "action-hmdb"];
const MODEL_LABEL: Record<string, string> = {
  yolo26n: "nano", yolo26s: "small", yolo26m: "medium", yolo26l: "large",
  "yolo26n-pose": "pose", action: "actions (NTU)", "action-hmdb": "actions (HMDB)",
};
const STRIDE_OPTS = [0, 1, 2, 3, 4];
const SPC_OPTS = [0, 1, 2, 4, 6, 12, 24];
const isActionKey = (m: string) => m.startsWith("action");

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

// The detections map for a given slot: shared object-detection lives in v.detections
// always; action keys come from v.detections (slot 1) or the slot's analysis bundle.
function detForModel(v: SavedVideo, model: string, slot: number) {
  if (!isActionKey(model)) return v.detections?.[model];
  return slot <= 1 ? v.detections?.[model] : v.analyses?.[slot]?.detections?.[model];
}

function AnalysisCard({ v, model, slot, roomName }: { v: SavedVideo; model: string; slot: number; roomName: string }) {
  const qc = useQueryClient();
  const d = detForModel(v, model, slot);
  const isPose = model.includes("pose");
  const isAction = isActionKey(model);
  const actionVariant = model === "action-hmdb" ? "hmdb" : "ntu";
  const hasOverlay = Boolean(d?.hasAnnotated && d.annotatedRelPath);
  const [overlay, setOverlay] = useState(true);
  const [drawer, setDrawer] = useState(false);
  const showOverlay = overlay && hasOverlay;
  const src = showOverlay ? fileUrl(d!.annotatedRelPath!) : fileUrl(v.relPath);
  const analyzing = clipAnalyzing(v);

  const reanalyze = useMutation({
    mutationFn: () => {
      if (!isAction) return post("/api/detect", { relPath: v.relPath, force: true });
      if (slot <= 1) return post("/api/action", { relPath: v.relPath, force: true, variant: actionVariant });
      // Re-run this slot (re-runs the whole recording for the slot's settings).
      const cfg = v.analyses?.[slot]?.config;
      return post("/api/analysis-slots", {
        day: v.day, rec: v.rec, slot,
        settings: cfg?.settings ?? {},
        variants: cfg?.variants ?? [actionVariant],
        disabled: {
          action: (cfg?.action as { disabled?: string[] })?.disabled ?? [],
          "action-hmdb": (cfg?.["action-hmdb"] as { disabled?: string[] })?.disabled ?? [],
        },
      });
    },
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

function Opt({ on, label, val, set }: { on: number; label: string; val: number[]; set: (n: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-bold text-muted">{label}</span>
      <div className="flex overflow-hidden rounded-lg border border-line text-xs font-bold">
        {val.map((n) => (
          <button key={n} onClick={() => set(n)} className={`px-2 py-1 ${n === on ? "bg-emerald-500 text-white" : "text-muted hover:bg-background"}`}>
            {n === 0 ? "auto" : n}
          </button>
        ))}
      </div>
    </div>
  );
}

// One recording's row: numbered slot tabs (sequential ordinals) + a New-analysis
// button that uses the shared sticky settings + a Download. Per recording.
function SlotTabs({
  session, selected, onSelect, model, stride, spc, disabled,
}: {
  session: Session;
  selected: number;
  onSelect: (slot: number) => void;
  model: string;
  stride: number;
  spc: number;
  disabled: Record<string, string[]>;
}) {
  const qc = useQueryClient();
  const { day, rec } = session.clips[0];
  const realSlots = [...new Set([1, ...session.clips.flatMap((c) => Object.keys(c.analyses ?? {}).map(Number))])]
    .sort((a, b) => a - b);
  const variant = model === "action-hmdb" ? "hmdb" : "ntu";

  const del = useMutation({
    mutationFn: () =>
      fetch(`/api/analysis-slots?path=${encodeURIComponent(`${day}/${rec}`)}&slot=${selected}`, { method: "DELETE" }),
    onSuccess: () => { onSelect(1); qc.invalidateQueries({ queryKey: ["saved"] }); },
  });
  const create = useMutation({
    mutationFn: async () => {
      const res = await fetch("/api/analysis-slots", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ day, rec, settings: { stride, samplesPerClassify: spc }, variants: [variant], disabled }),
      });
      return res.ok ? ((await res.json())?.slot as number | undefined) : undefined;
    },
    onSuccess: (slot) => { pingSavedSoon(qc); if (slot) onSelect(slot); },
  });

  return (
    <div className="mb-2 flex flex-wrap items-center gap-3">
      <span className="font-mono text-sm font-bold text-muted">{session.label}</span>
      <span className="rounded-full bg-card px-2 py-0.5 text-xs font-bold text-muted">
        {session.clips.length} cam{session.clips.length > 1 ? "s" : ""}
      </span>
      <div className="flex overflow-hidden rounded-lg border border-line text-xs font-bold">
        {realSlots.map((real, i) => (
          <button
            key={real}
            onClick={() => onSelect(real)}
            className={`px-2.5 py-1 ${real === selected ? "bg-foreground text-background" : "text-muted hover:bg-card"}`}
          >
            {i === 0 ? "1 · original" : i + 1}
          </button>
        ))}
      </div>
      {selected > 1 && (
        <button onClick={() => del.mutate()} disabled={del.isPending} title="Delete this analysis" className="rounded-lg border border-line p-1.5 text-rose-500 hover:bg-rose-50 disabled:opacity-50">
          <Trash2 className="size-3.5" />
        </button>
      )}
      <button
        onClick={() => create.mutate()}
        disabled={create.isPending}
        title={`Run a new ${variant.toUpperCase()} analysis on this recording with the chosen settings (creates a new tab)`}
        className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-bold text-white hover:bg-emerald-600 disabled:opacity-50"
      >
        {create.isPending ? <Loader2 className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />} New analysis ({variant.toUpperCase()})
      </button>
      <a
        href={`/api/saved/archive?path=${encodeURIComponent(`${day}/${rec}`)}`}
        title="Download this whole recording folder (both cameras + all analyses) as a .zip"
        className="ml-auto flex items-center gap-1 rounded-lg border border-line px-2 py-1 text-[11px] font-bold text-muted hover:bg-card"
      >
        <Download className="size-3.5" /> Download folder
      </a>
    </div>
  );
}

export function Analytics({ nodes: config }: { nodes: NodeConfig[] }) {
  const qc = useQueryClient();
  const saved = useSaved();
  const videos = saved.data?.videos ?? [];
  const analyzing = analyzingCount(saved.data);
  const nameByNode = new Map(config.map((n) => [n.id, n.name]));
  const sessions = groupSessions(videos);

  // Model toggle availability = union of shared-detection + action keys across all
  // slots, so an action model appears whenever any slot produced it.
  const available = MODEL_ORDER.filter((m) =>
    videos.some((v) => v.detections?.[m] || Object.values(v.analyses ?? {}).some((a) => a.detections?.[m])),
  );
  const [sel, setSel] = useState<string | null>(null);
  const model = sel && available.includes(sel) ? sel : (available[0] ?? "yolo26n");
  const [view, setView] = useState<"models" | "geometric">("models");
  const [slotBySession, setSlotBySession] = useState<Record<string, number>>({});

  // Shared, STICKY new-analysis settings — they persist until you change them
  // (no auto-reset). New analysis on any recording uses these.
  const [stride, setStride] = useState(0);
  const [spc, setSpc] = useState(0);
  const { data: globalCfg } = useQuery({
    queryKey: ["action-classes"],
    queryFn: async () => (await fetch("/api/action-classes", { cache: "no-store" })).json(),
    staleTime: Infinity,
  });
  const newDisabled = {
    action: globalCfg?.action?.disabled ?? [],
    "action-hmdb": globalCfg?.["action-hmdb"]?.disabled ?? [],
  };

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
      ) : videos.length === 0 ? (
        <p className="text-sm text-muted">No clips to analyze yet — “Beam to laptop” on the Live tab first.</p>
      ) : (
        <>
          {/* top controls: model + the single sticky settings bar */}
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
            <Opt on={stride} label="stride" val={STRIDE_OPTS} set={setStride} />
            <Opt on={spc} label="samples / classify" val={SPC_OPTS} set={setSpc} />
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={() => detectAll.mutate()}
                disabled={analyzing > 0}
                className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
              >
                {analyzing > 0 ? <Loader2 className="size-4 animate-spin" /> : <ScanEye className="size-4" />}
                {analyzing > 0 ? `Analyzing ${analyzing}…` : "Re-analyze all"}
              </button>
              <button onClick={() => actionAll.mutate()} disabled={analyzing > 0 || actionAll.isPending} title="Per-person actions on all clips (ST-GCN++ / NTU-RGB+D 60)" className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50">
                <Video className="size-4" /> Actions (NTU)
              </button>
              <button onClick={() => actionAllHmdb.mutate()} disabled={analyzing > 0 || actionAllHmdb.isPending} title="Per-person actions on all clips (PoseC3D / HMDB51)" className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50">
                <Video className="size-4" /> Actions (HMDB)
              </button>
              {analyzing > 0 && (
                <button onClick={() => cancel.mutate()} disabled={cancel.isPending} className="flex items-center gap-1.5 rounded-xl bg-rose-200 px-3 py-1.5 text-sm font-bold text-rose-800 hover:bg-rose-300">
                  <X className="size-4" /> Cancel
                </button>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-7">
            {sessions.map((s) => {
              const slot = slotBySession[s.key] ?? 1;
              return (
                <div key={s.key}>
                  <SlotTabs
                    session={s}
                    selected={slot}
                    model={model}
                    stride={stride}
                    spc={spc}
                    disabled={newDisabled}
                    onSelect={(n) => setSlotBySession((m) => ({ ...m, [s.key]: n }))}
                  />
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {s.clips.map((v) => (
                      <AnalysisCard key={v.relPath} v={v} model={model} slot={slot} roomName={nameByNode.get(v.node) ?? v.node} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
