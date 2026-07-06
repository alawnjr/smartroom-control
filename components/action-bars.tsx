"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { barClass, strokeColor } from "@/lib/action-colors";

// Per-window classifier output recorded by detect/action.py (camera_main.actions.<model>.json).
type Entry = { t: number; action: string; conf: number; kept?: boolean; top?: [string, number][] };
type Timeline = { tracks: Record<string, Entry[]> };

// How many top classes to show as bars, and the confidence a full-width bar maps
// to (lower = longer bars). These heads rarely exceed ~0.4 even when confident.
const TOP_N = 5;
const SCALE = 0.4;

function actionsPath(relPath: string, model: string) {
  return relPath.replace(/\.mp4$/, `.actions.${model}.json`);
}

function entryTop(e: Entry): [string, number][] {
  return e.top ?? [[e.action, e.conf]];
}

// Interpolate the class distribution at the playhead by blending the two windows
// straddling it (linear in their probabilities), so the bars glide between
// windows instead of snapping at each ~0.4s classification. Returns the blended
// label->prob map plus the nearer window (for the header label / idle state).
function distAt(entries: Entry[], t: number): { dist: Map<string, number>; ref: Entry } | null {
  let i = -1;
  for (let k = 0; k < entries.length; k++) {
    if (entries[k].t <= t) i = k;
    else break;
  }
  if (i < 0) return null; // before the first window
  const prev = entries[i];
  const next = entries[i + 1];
  const frac = next && next.t > prev.t ? Math.min(1, Math.max(0, (t - prev.t) / (next.t - prev.t))) : 0;

  const dist = new Map<string, number>();
  const add = (pairs: [string, number][], w: number) => {
    for (const [label, p] of pairs) dist.set(label, (dist.get(label) ?? 0) + p * w);
  };
  add(entryTop(prev), 1 - frac);
  if (next) add(entryTop(next), frac);
  return { dist, ref: frac >= 0.5 && next ? next : prev };
}

// Probability-over-time line graph for one person: a line per chip-action (same
// colors as the bars), with labelled axes, gridlines, a legend, and a playhead
// that tracks the video. Static geometry is memoised; only the playhead moves
// each frame. A class's probability is read from each window's top-K (0 when it
// isn't in the top-K that window).
const LW = 320;
const LH = 150;
const PAD = { l: 30, r: 8, t: 8, b: 26 };

function ActionLines({ entries, actions, currentTime }: { entries: Entry[]; actions: string[]; currentTime: number }) {
  const plot = useMemo(() => {
    if (entries.length < 2) return null;
    const tMax = entries[entries.length - 1].t || 1;
    const probAt = (e: Entry, label: string) => entryTop(e).find(([l]) => l === label)?.[1] ?? 0;
    // Only plot labels that meaningfully show up (keeps it to a few lines).
    const labels = actions.filter((a) => entries.some((e) => probAt(e, a) > 0.08)).slice(0, 5);
    if (labels.length === 0) return null;
    let pMax = 0.3;
    for (const a of labels) for (const e of entries) pMax = Math.max(pMax, probAt(e, a));
    pMax = Math.ceil(pMax * 10) / 10; // round up to a tidy 0.1

    const x0 = PAD.l;
    const x1 = LW - PAD.r;
    const y0 = LH - PAD.b;
    const y1 = PAD.t;
    const x = (t: number) => x0 + (t / tMax) * (x1 - x0);
    const y = (p: number) => y0 + (1 - p / pMax) * (y1 - y0);
    const lines = labels.map((a) => ({
      label: a,
      color: strokeColor(a, actions),
      points: entries.map((e) => `${x(e.t).toFixed(1)},${y(probAt(e, a)).toFixed(1)}`).join(" "),
    }));
    const yTicks = [0, pMax / 2, pMax];
    const xTicks = [0, tMax / 2, tMax];
    return { tMax, pMax, x, y, x0, x1, y0, y1, lines, yTicks, xTicks };
  }, [entries, actions]);

  if (!plot) return null;
  const px = plot.x(Math.min(currentTime, plot.tMax));

  return (
    <div className="mt-1.5">
      <div className="mb-1 flex flex-wrap gap-x-3 gap-y-0.5">
        {plot.lines.map((l) => (
          <span key={l.label} className="flex items-center gap-1 text-[10px] text-muted">
            <span className="h-0.5 w-3 rounded-full" style={{ background: l.color }} />
            {l.label}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${LW} ${LH}`} className="w-full" role="img" aria-label="action probability over time">
        {/* y gridlines + ticks (probability) */}
        {plot.yTicks.map((p) => (
          <g key={p}>
            <line x1={plot.x0} x2={plot.x1} y1={plot.y(p)} y2={plot.y(p)} stroke="rgb(0 0 0 / 0.08)" strokeWidth={1} />
            <text x={plot.x0 - 4} y={plot.y(p) + 3} fontSize={9} textAnchor="end" fill="rgb(0 0 0 / 0.4)">
              {p.toFixed(1)}
            </text>
          </g>
        ))}
        {/* x ticks (seconds) */}
        {plot.xTicks.map((t) => (
          <text key={t} x={plot.x(t)} y={plot.y0 + 13} fontSize={9} textAnchor="middle" fill="rgb(0 0 0 / 0.4)">
            {t.toFixed(t < 10 ? 1 : 0)}
          </text>
        ))}
        {/* axes */}
        <line x1={plot.x0} x2={plot.x1} y1={plot.y0} y2={plot.y0} stroke="rgb(0 0 0 / 0.25)" strokeWidth={1} />
        <line x1={plot.x0} x2={plot.x0} y1={plot.y1} y2={plot.y0} stroke="rgb(0 0 0 / 0.25)" strokeWidth={1} />
        {/* axis titles */}
        <text x={(plot.x0 + plot.x1) / 2} y={LH - 2} fontSize={9} textAnchor="middle" fill="rgb(0 0 0 / 0.5)">time (s)</text>
        <text x={9} y={(plot.y0 + plot.y1) / 2} fontSize={9} textAnchor="middle" fill="rgb(0 0 0 / 0.5)" transform={`rotate(-90 9 ${(plot.y0 + plot.y1) / 2})`}>probability</text>
        {/* series */}
        {plot.lines.map((l) => (
          <polyline key={l.label} points={l.points} fill="none" stroke={l.color} strokeWidth={1.75} strokeLinejoin="round" />
        ))}
        {/* playhead */}
        <line x1={px} x2={px} y1={plot.y1} y2={plot.y0} stroke="rgb(0 0 0 / 0.5)" strokeWidth={1} strokeDasharray="3 2" />
      </svg>
    </div>
  );
}

// Live per-person "most confident classes" bar graph, synced to the playing
// video's currentTime. One panel per tracked person; bars are the top-N classes
// for the window covering the current instant, colored to match the action chips.
// Reads the action sidecar directly via /api/saved/file (static once analyzed).
export function ActionBars({
  relPath,
  model,
  currentTime,
  actions,
}: {
  relPath: string;
  model: string;
  currentTime: number;
  actions: string[];
}) {
  const path = actionsPath(relPath, model);
  const { data } = useQuery({
    queryKey: ["action-timeline", path],
    queryFn: async (): Promise<Timeline | null> => {
      const res = await fetch(`/api/saved/file?path=${encodeURIComponent(path)}`, { cache: "force-cache" });
      if (!res.ok) return null;
      try {
        return await res.json();
      } catch {
        return null;
      }
    },
    staleTime: 5 * 60_000,
  });

  const tracks = data?.tracks;
  const ids = useMemo(
    () => (tracks ? Object.keys(tracks).sort((a, b) => Number(a) - Number(b)) : []),
    [tracks],
  );
  if (!tracks || ids.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      {ids.map((id) => {
        const at = distAt(tracks[id], currentTime);
        const idle = !at || at.ref.kept === false;
        const bars = at ? [...at.dist.entries()].sort((a, b) => b[1] - a[1]).slice(0, TOP_N) : [];

        return (
          <div key={id} className="rounded-xl border border-line bg-background p-2">
            <div className="mb-1 flex items-center justify-between text-xs font-bold">
              <span>#{id}</span>
              <span className={idle ? "text-muted" : "text-emerald-600"}>{idle ? "idle" : at?.ref.action}</span>
            </div>
            <div className="flex flex-col gap-1">
              {bars.length === 0 ? (
                <div className="text-[10px] text-muted">—</div>
              ) : (
                bars.map(([label, p]) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <span className="w-24 shrink-0 truncate text-[10px] text-muted" title={label}>
                      {label}
                    </span>
                    <div className="relative h-2.5 flex-1 overflow-hidden rounded bg-line/40">
                      <div
                        className={`absolute inset-y-0 left-0 rounded transition-[width] duration-100 ease-linear ${barClass(label, actions)}`}
                        style={{ width: `${Math.min(100, Math.max(2, (p / SCALE) * 100))}%` }}
                      />
                    </div>
                    <span className="w-8 shrink-0 text-right font-mono text-[10px] text-muted">{p.toFixed(2)}</span>
                  </div>
                ))
              )}
            </div>
            <ActionLines entries={tracks[id]} actions={actions} currentTime={currentTime} />
          </div>
        );
      })}
    </div>
  );
}
