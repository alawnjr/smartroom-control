"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, LineChart, Loader2, RefreshCw, ScanEye, ShieldAlert, ShieldCheck, Video, X } from "lucide-react";

import { ClipAnalyticsDrawer } from "@/components/clip-analytics-drawer";
import { GeometricPanel } from "@/components/geometric-page";
import { OccupancyGraph } from "@/components/occupancy-graph";
import { tagClass } from "@/lib/action-colors";
import { analyzingCount, clipAnalyzing, groupSessions, pingSavedSoon, useSaved, validatingCount, type Session } from "@/lib/use-saved";
import type { DetectionSummary, NodeConfig, SavedVideo } from "@/lib/types";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose", "action", "action-hmdb"];
const MODEL_LABEL: Record<string, string> = {
  yolo26n: "nano", yolo26s: "small", yolo26m: "medium", yolo26l: "large",
  "yolo26n-pose": "pose", action: "actions (NTU)", "action-hmdb": "actions (HMDB)",
};
const STRIDE_OPTS = [0, 1, 2, 3, 4];
const SPC_OPTS = [0, 1, 2, 4, 6, 12, 24];
const POSE_OPTS = [
  { val: "yolo", label: "YOLO" },
  { val: "rtmpose", label: "RTM" },
] as const;
type PoseSource = (typeof POSE_OPTS)[number]["val"];
const isActionKey = (m: string) => m.startsWith("action");

function fileUrl(relPath: string, version?: number) {
  const base = `/api/saved/file?path=${encodeURIComponent(relPath)}`;
  return version ? `${base}&v=${version}` : base;
}
function post(url: string, body?: unknown) {
  return fetch(url, {
    method: "POST",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).catch(() => {});
}

// Data-validation state for one clip (v.validation, from detect/validate.py).
// Green = every check passed; red = some failed — clicking the red chip toggles
// the failure-detail panel (onToggle). Sits next to the per-card revalidate button.
function ValidationChip({ v, onToggle }: { v: SavedVideo; onToggle: () => void }) {
  const val = v.validation;
  if (!val) return null;
  if (val.status === "analyzing")
    return <span className="rounded-md bg-neutral-200 px-1.5 py-0.5 text-[10px] font-bold text-neutral-600 dark:bg-neutral-700 dark:text-neutral-300">validating…</span>;
  if (val.status === "error")
    return <span title={val.error} className="rounded-md bg-rose-100 px-1.5 py-0.5 text-[10px] font-bold text-rose-700">validation error</span>;
  if (val.status !== "done") return null;
  if ((val.failed ?? 0) > 0)
    return (
      <button
        onClick={onToggle}
        title="Click for details"
        className="flex cursor-pointer items-center gap-1 rounded-md bg-rose-100 px-1.5 py-0.5 text-[10px] font-bold text-rose-700 hover:bg-rose-200"
      >
        <ShieldAlert className="size-3" /> {val.failed} failed
      </button>
    );
  return (
    <button
      onClick={onToggle}
      title={`All ${val.passed} checks passed — click for the list`}
      className="flex cursor-pointer items-center gap-1 rounded-md bg-emerald-100 px-1.5 py-0.5 text-[10px] font-bold text-emerald-700 hover:bg-emerald-200"
    >
      <ShieldCheck className="size-3" /> valid
    </button>
  );
}

// The expanded per-check panel: EVERY check with its outcome and the
// human-readable detail from validate.py (e.g. "404 rows vs 404 video frames"),
// failures first. Rendered inside the card when the chip is clicked.
function ValidationDetails({ v }: { v: SavedVideo }) {
  const val = v.validation;
  if (!val || val.status !== "done") return null;
  const total = (val.passed ?? 0) + (val.failed ?? 0);
  const allOk = (val.failed ?? 0) === 0;
  const checks = [...(val.checks ?? [])].sort((a, b) => Number(a.ok) - Number(b.ok));
  return (
    <div
      className={`mt-2 rounded-xl border p-2.5 ${
        allOk
          ? "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40"
          : "border-rose-200 bg-rose-50 dark:border-rose-900 dark:bg-rose-950/40"
      }`}
    >
      <div className={`mb-1.5 text-[11px] font-extrabold ${allOk ? "text-emerald-700 dark:text-emerald-300" : "text-rose-700 dark:text-rose-300"}`}>
        {allOk ? `all ${total} checks passed` : `${val.failed} of ${total} checks failed`}
      </div>
      <ul className="flex flex-col gap-1.5">
        {checks.map((c) => (
          <li key={c.name} className="flex items-start gap-1.5 text-[11px] leading-snug">
            {c.ok ? (
              <ShieldCheck className="mt-px size-3 shrink-0 text-emerald-600" />
            ) : (
              <ShieldAlert className="mt-px size-3 shrink-0 text-rose-600" />
            )}
            <span>
              <span className={`font-mono font-bold ${c.ok ? "text-emerald-700 dark:text-emerald-300" : "text-rose-700 dark:text-rose-300"}`}>
                {c.name.replaceAll("_", " ")}
              </span>
              <span className="text-foreground/70"> — {c.detail}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// One clip's card for the selected model. Object-detection and action results all
// live in-place next to the clip (v.detections[model]); re-analysis reruns the
// current model on this clip with the global settings shown in the bar above.
function AnalysisCard({ v, model, roomName, selected, onToggleSelect }: { v: SavedVideo; model: string; roomName: string; selected: boolean; onToggleSelect: () => void }) {
  const qc = useQueryClient();
  const d = v.detections?.[model];
  const isPose = model.includes("pose");
  const isAction = isActionKey(model);
  const actionVariant = model === "action-hmdb" ? "hmdb" : "ntu";
  const hasOverlay = Boolean(d?.hasAnnotated && d.annotatedRelPath);
  const [overlay, setOverlay] = useState(true);
  const [drawer, setDrawer] = useState(false);
  const [showValidation, setShowValidation] = useState(false);
  const showOverlay = overlay && hasOverlay;
  const src = showOverlay ? fileUrl(d!.annotatedRelPath!, d!.version) : fileUrl(v.relPath);
  const analyzing = clipAnalyzing(v);

  const reanalyze = useMutation({
    mutationFn: () =>
      isAction
        ? post("/api/action", { relPath: v.relPath, force: true, variant: actionVariant })
        : post("/api/detect", { relPath: v.relPath, force: true }),
    onSuccess: () => pingSavedSoon(qc),
  });
  const revalidate = useMutation({
    mutationFn: () => post("/api/validate", { relPath: v.relPath, force: true }),
    onSuccess: () => pingSavedSoon(qc),
  });

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <label className="flex cursor-pointer items-center gap-2 text-sm font-extrabold" title="Select for re-analysis">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="size-4 cursor-pointer accent-emerald-500"
          />
          {roomName} <span className="font-mono text-xs font-normal text-muted">· {v.rec.split("_").pop()}</span>
        </label>
        <div className="flex items-center gap-2">
          <ValidationChip v={v} onToggle={() => setShowValidation((s) => !s)} />
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
          <button
            onClick={() => revalidate.mutate()}
            disabled={revalidate.isPending || v.validation?.status === "analyzing"}
            title="Re-run data validation on this clip"
            className="rounded-lg border border-line p-1.5 text-muted hover:bg-background disabled:opacity-50"
          >
            <ShieldCheck className={`size-3.5 ${revalidate.isPending || v.validation?.status === "analyzing" ? "animate-pulse" : ""}`} />
          </button>
        </div>
      </div>

      {showValidation && <div className="mb-2 -mt-1"><ValidationDetails v={v} /></div>}

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

// String-valued sibling of Opt: the skeleton source feeding the action classifier.
function PoseOpt({ on, set }: { on: PoseSource; set: (p: PoseSource) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-bold text-muted">pose</span>
      <div className="flex overflow-hidden rounded-lg border border-line text-xs font-bold">
        {POSE_OPTS.map((o) => (
          <button key={o.val} onClick={() => set(o.val)} className={`px-2 py-1 ${o.val === on ? "bg-emerald-500 text-white" : "text-muted hover:bg-background"}`}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// The settings that produced the current model's analysis for this recording, read
// back from the sidecar (both cameras run with the same settings). Action models
// only — detection models have no stride/pose knobs.
function runSettingsText(model: string, d?: DetectionSummary): string | null {
  if (!isActionKey(model) || !d || d.status !== "done" || d.stride == null) return null;
  const pose = d.poseSource === "rtmpose" ? "RTMPose" : "YOLO pose";
  const spc = d.samplesPerClassify != null ? `${d.samplesPerClassify} samples/classify` : null;
  return [`stride ${d.stride}`, spc, pose].filter(Boolean).join(" · ");
}

// One recording's header: a select-all checkbox, label, camera count, the settings
// the current model ran with, and a download-folder link.
function SessionHeader({ session, allSelected, onToggle, model, summary }: {
  session: Session;
  allSelected: boolean;
  onToggle: () => void;
  model: string;
  summary?: DetectionSummary;
}) {
  const { day, rec } = session.clips[0];
  const settings = runSettingsText(model, summary);
  return (
    <div className="mb-2 flex flex-wrap items-center gap-3">
      <input
        type="checkbox"
        checked={allSelected}
        onChange={onToggle}
        title="Select all cameras in this recording"
        className="size-4 cursor-pointer accent-emerald-500"
      />
      <span className="font-mono text-sm font-bold text-muted">{session.label}</span>
      <span className="rounded-full bg-card px-2 py-0.5 text-xs font-bold text-muted">
        {session.clips.length} cam{session.clips.length > 1 ? "s" : ""}
      </span>
      {settings && (
        <span
          title={`Settings used for ${MODEL_LABEL[model] ?? model} on this recording`}
          className="rounded-full border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-xs font-bold text-emerald-700"
        >
          {settings}
        </span>
      )}
      <a
        href={`/api/saved/archive?path=${encodeURIComponent(`${day}/${rec}`)}`}
        title="Download this whole recording folder (both cameras) as a .zip"
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
  const validating = validatingCount(saved.data);
  // Validation outcome summary for the top bar: how many clips are clean vs flagged.
  const validated = (saved.data?.videos ?? []).filter((v) => v.validation?.status === "done");
  const validFailed = validated.filter((v) => (v.validation?.failed ?? 0) > 0).length;
  const nameByNode = new Map(config.map((n) => [n.id, n.name]));
  const sessions = groupSessions(videos);

  const available = MODEL_ORDER.filter((m) => videos.some((v) => v.detections?.[m]));
  const [sel, setSel] = useState<string | null>(null);
  const model = sel && available.includes(sel) ? sel : (available[0] ?? "yolo26n");
  const [view, setView] = useState<"models" | "geometric">("models");

  // Analysis settings are GLOBAL, stored in action-classes.json's `settings` block
  // (shared with detect/action.py). The bar edits them; a re-analysis picks them up.
  const { data: globalCfg } = useQuery({
    queryKey: ["action-classes"],
    queryFn: async () => (await fetch("/api/action-classes", { cache: "no-store" })).json(),
    staleTime: Infinity,
  });
  const cur = {
    stride: globalCfg?.settings?.stride ?? 0,
    spc: globalCfg?.settings?.samplesPerClassify ?? 0,
    poseSource: (globalCfg?.settings?.poseSource === "rtmpose" ? "rtmpose" : "yolo") as PoseSource,
  };
  // Optimistically patch the cached config for instant feedback, then persist. The
  // route accepts one field at a time ({stride} | {samplesPerClassify} | {poseSource}).
  const setSetting = (patch: Record<string, number | string>) => {
    qc.setQueryData(["action-classes"], (old: Record<string, unknown> | undefined) => ({
      ...(old ?? {}),
      settings: { ...((old?.settings as object) ?? {}), ...patch },
    }));
    post("/api/action-classes", patch);
  };

  // Checklist: which clips a batch re-analysis applies to. Empty = all clips.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const toggleSelect = (relPath: string) =>
    setSelected((s) => {
      const next = new Set(s);
      next.has(relPath) ? next.delete(relPath) : next.add(relPath);
      return next;
    });
  const allRelPaths = videos.map((v) => v.relPath);
  const selectedCount = allRelPaths.filter((r) => selected.has(r)).length; // ignores stale entries
  // Body for a batch run: the selected clips, or all when nothing is checked.
  const batchBody = () => (selectedCount > 0 ? { relPaths: allRelPaths.filter((r) => selected.has(r)), force: true } : { force: true });

  const detectAll = useMutation({ mutationFn: () => post("/api/detect", batchBody()), onSuccess: () => pingSavedSoon(qc) });
  const validateAll = useMutation({ mutationFn: () => post("/api/validate", batchBody()), onSuccess: () => pingSavedSoon(qc) });
  const actionAll = useMutation({ mutationFn: () => post("/api/action", { ...batchBody(), variant: "ntu" }), onSuccess: () => pingSavedSoon(qc) });
  const actionAllHmdb = useMutation({ mutationFn: () => post("/api/action", { ...batchBody(), variant: "hmdb" }), onSuccess: () => pingSavedSoon(qc) });
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
          {/* top controls: model picker + global analysis settings + batch actions */}
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
            <Opt on={cur.stride} label="stride" val={STRIDE_OPTS} set={(n) => setSetting({ stride: n })} />
            <Opt on={cur.spc} label="samples / classify" val={SPC_OPTS} set={(n) => setSetting({ samplesPerClassify: n })} />
            <PoseOpt on={cur.poseSource} set={(p) => setSetting({ poseSource: p })} />
            <div className="ml-auto flex items-center gap-2">
              {/* checklist scope: which clips the batch buttons act on */}
              <span className="text-xs font-bold text-muted">
                {selectedCount > 0 ? `${selectedCount} selected` : "all clips"}
              </span>
              <button
                onClick={() => setSelected(new Set(allRelPaths))}
                className="rounded-lg border border-line px-2 py-1 text-[11px] font-bold text-muted hover:bg-background"
              >
                Select all
              </button>
              <button
                onClick={() => setSelected(new Set())}
                disabled={selectedCount === 0}
                className="rounded-lg border border-line px-2 py-1 text-[11px] font-bold text-muted hover:bg-background disabled:opacity-40"
              >
                Clear
              </button>
              <button
                onClick={() => detectAll.mutate()}
                disabled={analyzing > 0}
                title="Object detection on the selected clips (or all if none selected)"
                className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
              >
                {analyzing > 0 ? <Loader2 className="size-4 animate-spin" /> : <ScanEye className="size-4" />}
                {analyzing > 0 ? `Analyzing ${analyzing}…` : selectedCount > 0 ? `Re-detect ${selectedCount}` : "Re-detect all"}
              </button>
              <button
                onClick={() => validateAll.mutate()}
                disabled={validateAll.isPending || validating > 0}
                title="Data-integrity checks (video, metadata, timestamps) on the selected clips, or all if none selected"
                className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50"
              >
                {validateAll.isPending || validating > 0 ? <Loader2 className="size-4 animate-spin" /> : <ShieldCheck className="size-4" />}
                {validating > 0
                  ? `Validating ${validating}…`
                  : validateAll.isPending
                    ? "Validating…"
                    : selectedCount > 0
                      ? `Validate ${selectedCount}`
                      : "Validate all"}
              </button>
              {validated.length > 0 && validating === 0 && (
                <span
                  title={validFailed > 0 ? "Some clips failed validation — see the red chips on their cards" : "All validated clips passed"}
                  className={`flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-bold ${
                    validFailed > 0 ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"
                  }`}
                >
                  {validFailed > 0 ? <ShieldAlert className="size-3.5" /> : <ShieldCheck className="size-3.5" />}
                  {validFailed > 0 ? `${validFailed}/${validated.length} flagged` : `${validated.length} valid`}
                </span>
              )}
              <button onClick={() => actionAll.mutate()} disabled={analyzing > 0 || actionAll.isPending} title="Per-person actions on the selected clips, or all if none selected (ST-GCN++ / NTU-RGB+D 60)" className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50">
                <Video className="size-4" /> Actions (NTU)
              </button>
              <button onClick={() => actionAllHmdb.mutate()} disabled={analyzing > 0 || actionAllHmdb.isPending} title="Per-person actions on the selected clips, or all if none selected (PoseC3D / HMDB51)" className="flex items-center gap-1.5 rounded-xl border border-line bg-card px-3 py-1.5 text-sm font-bold hover:bg-background disabled:opacity-50">
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
              const clipPaths = s.clips.map((c) => c.relPath);
              const allSel = clipPaths.every((r) => selected.has(r));
              const toggleSession = () =>
                setSelected((cur) => {
                  const next = new Set(cur);
                  if (allSel) clipPaths.forEach((r) => next.delete(r));
                  else clipPaths.forEach((r) => next.add(r));
                  return next;
                });
              // Prefer a camera whose sidecar actually carries the settings (older
              // runs predate the fields), so a mixed old/new pair still shows the chip.
              const dets = s.clips.map((c) => c.detections?.[model]).filter((d) => d?.status === "done");
              const summary = dets.find((d) => d?.stride != null) ?? dets[0];
              return (
                <div key={s.key}>
                  <SessionHeader session={s} allSelected={allSel} onToggle={toggleSession} model={model} summary={summary} />
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {s.clips.map((v) => (
                      <AnalysisCard
                        key={v.relPath}
                        v={v}
                        model={model}
                        roomName={nameByNode.get(v.node) ?? v.node}
                        selected={selected.has(v.relPath)}
                        onToggleSelect={() => toggleSelect(v.relPath)}
                      />
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
