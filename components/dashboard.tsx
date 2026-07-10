"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CloudUpload, Download, Minus, Plus } from "lucide-react";

import { ActionClassesPage } from "@/components/action-classes-page";
import { Analytics } from "@/components/analytics";
import { analyzingCount, clipAnalyzing, groupSessions, pingSavedSoon, useSaved } from "@/lib/use-saved";
import type { CombinedStatus, DetectionSummary, NodeConfig, NodeStatus, SavedVideo } from "@/lib/types";

// Per-room identity colours, cycled by node order.
const ROOM = [
  { bar: "bg-emerald-400", grad: "from-emerald-100 to-emerald-200/70", pill: "bg-emerald-200/80 text-emerald-900", dot: "bg-emerald-500", ring: "ring-emerald-300/60" },
  { bar: "bg-rose-400", grad: "from-rose-100 to-rose-200/70", pill: "bg-rose-200/80 text-rose-900", dot: "bg-rose-500", ring: "ring-rose-300/60" },
  { bar: "bg-sky-400", grad: "from-sky-100 to-sky-200/70", pill: "bg-sky-200/80 text-sky-900", dot: "bg-sky-500", ring: "ring-sky-300/60" },
  { bar: "bg-amber-400", grad: "from-amber-100 to-amber-200/70", pill: "bg-amber-200/80 text-amber-900", dot: "bg-amber-500", ring: "ring-amber-300/60" },
];
const TAG = ["bg-amber-200 text-amber-900", "bg-sky-200 text-sky-900", "bg-violet-200 text-violet-900", "bg-emerald-200 text-emerald-900", "bg-rose-200 text-rose-900"];

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function streamUrl(id: string) {
  // proxied through the app (same origin) so it works over a remote tunnel
  return `/api/stream/${encodeURIComponent(id)}`;
}
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

// "Sync to mirror" — pushes the saved recordings to the public Vercel mirror
// (incremental; POST /api/mirror spawns the mirror repo's sync script). Polls
// while a sync runs and shows the run's summary line when it finishes.
function MirrorButton() {
  const qc = useQueryClient();
  const status = useQuery<{
    running: boolean;
    summary: string | null;
    failed: boolean;
    finishedAt: number | null;
  }>({
    queryKey: ["mirror"],
    queryFn: () => fetch("/api/mirror").then((r) => r.json()),
    refetchInterval: (q) => (q.state.data?.running ? 2000 : false),
  });
  const sync = useMutation({
    mutationFn: () => post("/api/mirror"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["mirror"] }),
  });
  const running = status.data?.running ?? false;
  // "manifest: 12 sessions | uploaded 34 blobs (56.7 MB) | 890 up-to-date"
  const summary = status.data?.summary?.replace(/^manifest:\s*/, "") ?? null;
  return (
    <div className="flex items-center gap-2">
      {running ? (
        <Pill className="bg-sky-100 text-sky-700">
          <span className="size-2 animate-pulse rounded-full bg-sky-500" /> mirroring…
        </Pill>
      ) : status.data?.failed ? (
        <Pill className="bg-rose-100 text-rose-700">mirror sync failed</Pill>
      ) : summary ? (
        <span className="hidden text-xs text-muted sm:inline" title={summary}>
          {summary.split("|").slice(1).join("·").trim()}
        </span>
      ) : null}
      <button
        onClick={() => sync.mutate()}
        disabled={running || sync.isPending}
        className="flex items-center gap-1.5 rounded-full border border-line bg-card px-3 py-1 text-sm font-bold text-muted hover:bg-foreground hover:text-background disabled:opacity-50"
        title="Upload new recordings + inference to the public Vercel mirror"
      >
        <CloudUpload size={14} /> Sync to mirror
      </button>
    </div>
  );
}

function Pill({ className = "", children }: { className?: string; children: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold ${className}`}>
      {children}
    </span>
  );
}

// ---------- room card ----------
function RoomCard({ node, idx, onWake }: { node: NodeStatus; idx: number; onWake: () => void }) {
  const c = ROOM[idx % ROOM.length];
  const running = Boolean(node.online && node.status?.running);
  const offline = !node.online;
  const [k, setK] = useState(0);
  const pct = running && node.status?.duration ? Math.min(100, (node.status.elapsed / node.status.duration) * 100) : 0;

  return (
    <div className="overflow-hidden rounded-[26px] border border-line bg-card shadow-sm">
      <div className={`h-1.5 w-full ${offline ? "bg-neutral-300" : running ? "bg-rose-400" : c.bar}`} />
      <div className="p-3">
        <div
          className={`relative aspect-[16/10] w-full overflow-hidden rounded-2xl bg-gradient-to-br ${
            offline ? "from-neutral-200 to-neutral-300 stripes" : c.grad
          }`}
        >
          <span className="absolute left-3 top-3 z-10">
            {offline ? (
              <Pill className="bg-neutral-200/90 text-neutral-500">Off the grid</Pill>
            ) : running ? (
              <Pill className="bg-rose-100/90 text-rose-700">
                <span className="size-2 animate-pulse rounded-full bg-rose-500" /> Rolling · {fmt(node.status!.remaining)}
              </Pill>
            ) : (
              <Pill className="bg-white/80 text-emerald-700">
                <span className="size-2 rounded-full bg-emerald-500" /> Live
              </Pill>
            )}
          </span>

          {offline ? (
            <div className="flex h-full items-center justify-center text-sm font-semibold text-neutral-400">Napping</div>
          ) : running ? (
            <div className="flex h-full items-center justify-center text-sm font-semibold text-rose-400/70 scanlines">
              recording…
            </div>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={k}
              src={`${streamUrl(node.id)}?k=${k}`}
              alt={`${node.name} live`}
              className="h-full w-full object-cover scanlines"
              onError={() => setTimeout(() => setK((v) => v + 1), 1500)}
            />
          )}
          {running && (
            <div className="absolute inset-x-0 bottom-0 h-1.5 bg-rose-200">
              <div className="h-full bg-rose-500 transition-all" style={{ width: `${pct}%` }} />
            </div>
          )}
        </div>

        <div className="mt-3 flex items-end justify-between px-1">
          <div>
            <div className="text-lg font-extrabold leading-tight">{node.name}</div>
            <div className="font-mono text-[11px] text-muted">{node.host}</div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="flex gap-1">
              {offline
                ? [0, 1].map((i) => <span key={i} className="size-2 rounded-full bg-neutral-300" />)
                : [0, 1, 2].map((i) => <span key={i} className={`size-2 rounded-full ${c.dot}`} />)}
            </div>
            {offline ? (
              <button onClick={onWake} className="text-xs font-bold text-neutral-400 hover:text-neutral-600">
                wake it up →
              </button>
            ) : running ? (
              <span className="text-xs font-bold text-muted">on camera</span>
            ) : (
              <span className="text-xs font-bold text-muted">live</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- ready to roll ----------
type RunStat = {
  kind: string;
  label: string;
  finishedAt: string;
  elapsedSec: number;
  processed: number;
  errors?: number;
  perClipSec?: number | null;
};

function fmtDur(s: number) {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${Math.round(s % 60).toString().padStart(2, "0")}s`;
}

// Last-run stats for each analysis kind (detect / NTU / HMDB), shown in the sidebar.
function AnalyzeStats() {
  const { data } = useQuery({
    queryKey: ["analyze-stats"],
    queryFn: async () =>
      (await fetch("/api/analyze-stats", { cache: "no-store" })).json() as Promise<{ runs: RunStat[] }>,
    refetchInterval: 8000,
  });
  const runs = data?.runs ?? [];
  if (runs.length === 0) return null;
  return (
    <div className="mt-4 border-t border-line pt-3">
      <div className="text-xs font-bold text-muted">Last analysis runs</div>
      <div className="mt-1.5 flex flex-col gap-1.5">
        {runs.map((r) => (
          <div key={r.kind} className="flex items-center justify-between gap-2 text-xs">
            <span className="font-bold">{r.label}</span>
            <span className="text-right font-mono text-[11px] text-muted">
              {r.processed} clip{r.processed === 1 ? "" : "s"} · {fmtDur(r.elapsedSec)}
              {r.perClipSec ? ` · ${fmtDur(r.perClipSec)}/clip` : ""}
              {r.errors ? ` · ${r.errors} err` : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReadyToRoll({
  duration,
  setDuration,
  onRecord,
  onStop,
  onBeam,
  recording,
  busy,
}: {
  duration: number;
  setDuration: (n: number) => void;
  onRecord: () => void;
  onStop: () => void;
  onBeam: () => void;
  recording: boolean;
  busy: boolean;
}) {
  const step = (d: number) => setDuration(Math.max(5, Math.min(3600, duration + d)));
  return (
    <div className="rounded-[26px] border border-line bg-card p-5 shadow-sm">
      <div className="text-xl font-extrabold">Ready to roll?</div>
      <div className="mt-0.5 text-sm text-muted">Captures every online room in sync.</div>

      <div className="mt-4 text-sm font-bold text-muted">How long?</div>
      <div className="mt-1.5 flex items-stretch gap-2">
        <button onClick={() => step(-5)} className="flex w-12 items-center justify-center rounded-xl border border-line bg-background hover:bg-line">
          <Minus className="size-4" />
        </button>
        <div className="flex flex-1 items-center justify-center rounded-xl border border-line bg-background text-lg font-extrabold tabular-nums">
          {duration}s
        </div>
        <button onClick={() => step(5)} className="flex w-12 items-center justify-center rounded-xl border border-line bg-background hover:bg-line">
          <Plus className="size-4" />
        </button>
      </div>

      <button
        onClick={onRecord}
        disabled={busy}
        className="mt-3 w-full rounded-xl bg-emerald-500 py-3 font-extrabold text-white shadow-sm hover:bg-emerald-600 disabled:opacity-50"
      >
        Record everything
      </button>
      <button
        onClick={onStop}
        disabled={!recording}
        className="mt-2.5 w-full rounded-xl bg-rose-200 py-2.5 font-bold text-rose-800 hover:bg-rose-300 disabled:opacity-40"
      >
        Stop recording
      </button>
      <button
        onClick={onBeam}
        disabled={busy}
        className="mt-2.5 flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-card py-2.5 font-bold hover:bg-background disabled:opacity-50"
      >
        <Download className="size-4" /> Beam to laptop
      </button>
      <AnalyzeStats />
    </div>
  );
}

// ---------- highlight clip ----------
function ClipCard({ v, roomIdx }: { v: SavedVideo; roomIdx: number }) {
  const c = ROOM[roomIdx % ROOM.length];
  const qc = useQueryClient();
  const dets = v.detections ?? {};
  const detModels = Object.entries(dets).filter(([m]) => m.startsWith("yolo26") && !m.includes("pose"));
  const peak = Math.max(0, ...detModels.map(([, d]) => d.maxPersons ?? 0));
  const analyzed = Object.values(dets).some((d) => d.status === "done");
  const analyzing = clipAnalyzing(v);
  const action = dets.action;
  const tags = action?.status === "done" ? action.actions ?? [] : [];
  const dur = Math.max(0, ...Object.values(dets).map((d: DetectionSummary) => d.durationSec ?? 0));
  const recShort = `rec_${v.rec.split("_").pop() ?? v.rec}`;

  const analyze = useMutation({
    mutationFn: () => post("/api/detect", { relPath: v.relPath, force: true }),
    onSuccess: () => pingSavedSoon(qc),
  });

  return (
    <div className={`overflow-hidden rounded-[22px] border border-line bg-card shadow-sm ${analyzing ? `ring-2 ${c.ring}` : ""}`}>
      <div className={`relative aspect-[16/10] overflow-hidden bg-gradient-to-br ${analyzed ? c.grad : "from-neutral-200 to-neutral-300"}`}>
        <Pill className={`absolute left-2 top-2 z-10 px-2 py-0.5 text-[10px] ${analyzed ? "bg-white/70" : "bg-white/60 text-neutral-500"}`}>
          {roomName(roomIdx)}
        </Pill>
        {analyzed && (
          // eslint-disable-next-line @next/next/no-img-element
          <video preload="none" controls className="h-full w-full object-cover" src={fileUrl(v.relPath)} />
        )}
        {dur > 0 && (
          <span className="absolute bottom-2 right-2 rounded-md bg-black/55 px-1.5 py-0.5 font-mono text-[10px] text-white">
            {fmt(dur)}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between px-3 pb-1.5 pt-2">
        <div className="font-extrabold">{recShort}</div>
        {analyzed ? (
          <div className="flex items-center gap-1.5">
            <div className="flex gap-1">
              {Array.from({ length: Math.min(peak, 4) }).map((_, i) => (
                <span key={i} className={`size-2 rounded-full ${c.dot}`} />
              ))}
            </div>
            <span className="text-xs font-bold text-muted">peak {peak}</span>
          </div>
        ) : (
          <button
            onClick={() => analyze.mutate()}
            disabled={analyze.isPending || analyzing}
            className="rounded-full bg-amber-300 px-2.5 py-1 text-xs font-bold text-amber-900 hover:bg-amber-400 disabled:opacity-60"
          >
            {analyzing ? "analyzing…" : "Analyze me!"}
          </button>
        )}
      </div>
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pb-3">
          {tags.slice(0, 3).map((t, i) => (
            <span key={t} className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${TAG[i % TAG.length]}`}>
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function roomName(idx: number) {
  return `Smartroom ${idx + 1}`;
}

// ---------- dashboard ----------
export function Dashboard({ nodes: config }: { nodes: NodeConfig[] }) {
  const qc = useQueryClient();
  const [duration, setDuration] = useState(30);
  const [tab, setTab] = useState<"live" | "analytics" | "classes">("live");
  const [clock, setClock] = useState("");
  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  const status = useQuery({
    queryKey: ["status"],
    queryFn: async (): Promise<CombinedStatus> => (await fetch("/api/status", { cache: "no-store" })).json(),
    refetchInterval: 1000,
    refetchOnWindowFocus: false,
  });
  const saved = useSaved();

  const rooms: NodeStatus[] =
    status.data?.nodes ?? config.map((n) => ({ ...n, online: false, status: null }));
  const liveCount = rooms.filter((n) => n.online).length;
  const rollingCount = rooms.filter((n) => n.online && n.status?.running).length;
  const anyRolling = rollingCount > 0;
  const analyzing = analyzingCount(saved.data);

  const idxByNode = useMemo(() => {
    const m = new Map<string, number>();
    config.forEach((n, i) => m.set(n.id, i));
    return m;
  }, [config]);

  const record = useMutation({
    mutationFn: () => post("/api/record", { duration }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
  const stop = useMutation({
    mutationFn: () => post("/api/cancel"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
  const beam = useMutation({
    mutationFn: () => post("/api/save-all"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["saved"] });
      post("/api/detect");
      pingSavedSoon(qc);
    },
  });
  const busy = record.isPending || beam.isPending;

  const clips = saved.data?.videos ?? [];

  return (
    <div className="mx-auto w-full max-w-6xl px-5 pb-16">
      {/* header */}
      <header className="flex items-center justify-between py-5">
        <div className="flex items-center gap-3">
          <span className="flex size-7 items-center justify-center rounded-lg bg-emerald-500">
            <span className="size-2.5 rounded-full bg-white" />
          </span>
          <span className="text-xl font-extrabold">Smartroom</span>
          <div className="ml-2 flex overflow-hidden rounded-full border border-line">
            {(["live", "analytics", "classes"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-3.5 py-1 text-sm font-bold capitalize ${tab === t ? "bg-foreground text-background" : "text-muted hover:bg-card"}`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2.5">
          <Pill className="border border-emerald-300 bg-emerald-50 text-emerald-700">{liveCount} rooms live</Pill>
          {rollingCount > 0 && (
            <Pill className="bg-rose-100 text-rose-700">
              <span className="size-2 animate-pulse rounded-full bg-rose-500" /> {rollingCount} rolling
            </Pill>
          )}
          <span className="font-mono text-sm font-bold text-muted">{clock}</span>
        </div>
      </header>

      {tab === "analytics" ? (
        <Analytics nodes={config} />
      ) : tab === "classes" ? (
        <ActionClassesPage />
      ) : (
        <>
      {/* rooms + ready to roll */}
      <h2 className="mb-3 text-lg font-extrabold">The rooms right now</h2>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:col-span-2">
          {rooms.map((n, i) => (
            <RoomCard key={n.id} node={n} idx={i} onWake={() => status.refetch()} />
          ))}
        </div>
        <ReadyToRoll
          duration={duration}
          setDuration={setDuration}
          onRecord={() => record.mutate()}
          onStop={() => stop.mutate()}
          onBeam={() => beam.mutate()}
          recording={anyRolling}
          busy={busy}
        />
      </div>

      {/* highlight reel */}
      <div className="mb-3 mt-10 flex items-center justify-between">
        <h2 className="text-lg font-extrabold">
          Your highlight reel <span className="text-amber-500">✦</span>
        </h2>
        <div className="flex items-center gap-2">
          {analyzing > 0 && (
            <Pill className="bg-amber-100 text-amber-700">
              <span className="size-2 animate-pulse rounded-full bg-amber-500" /> analyzing {analyzing}
            </Pill>
          )}
          <MirrorButton />
          <Pill className="border border-line bg-card text-muted">{clips.length} clips</Pill>
        </div>
      </div>
      {clips.length === 0 ? (
        <p className="text-sm text-muted">No clips yet — hit “Beam to laptop” to pull recordings.</p>
      ) : (
        <div className="flex flex-col gap-7">
          {groupSessions(clips).map((s) => (
            <div key={s.key}>
              <div className="mb-2 flex items-center gap-2 text-sm font-bold text-muted">
                <span className="font-mono">{s.label}</span>
                <span className="rounded-full bg-card px-2 py-0.5 text-xs">
                  {s.clips.length} cam{s.clips.length > 1 ? "s" : ""}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {s.clips.map((v) => (
                  <ClipCard key={v.relPath} v={v} roomIdx={idxByNode.get(v.node) ?? 0} />
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
