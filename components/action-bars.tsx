"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

// Per-window classifier output recorded by detect/action.py (camera_main.actions.<model>.json).
type Entry = { t: number; action: string; conf: number; kept?: boolean; top?: [string, number][] };
type Timeline = { tracks: Record<string, Entry[]> };

// Top-N classes get their own slice; everything else collapses into "other".
const TOP_N = 3;
const COLORS = ["#10b981", "#0ea5e9", "#a855f7"]; // emerald / sky / violet
const OTHER_COLOR = "#d1d5db";

function actionsPath(relPath: string, model: string) {
  return relPath.replace(/\.mp4$/, `.actions.${model}.json`);
}

function entryTop(e: Entry): [string, number][] {
  return e.top ?? [[e.action, e.conf]];
}

// Interpolate the class distribution at the playhead by blending the two windows
// straddling it (linear in their probabilities). This makes the pie glide between
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

function polar(cx: number, cy: number, r: number, a: number) {
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

// SVG path for a pie wedge from angle a0 to a1 (radians, 0 = 3 o'clock).
function wedge(cx: number, cy: number, r: number, a0: number, a1: number) {
  const p0 = polar(cx, cy, r, a0);
  const p1 = polar(cx, cy, r, a1);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)} Z`;
}

type Slice = { label: string; value: number; color: string };

function Pie({ slices }: { slices: Slice[] }) {
  const S = 72;
  const c = S / 2;
  const r = S / 2 - 1;
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;
  // A single ~100% slice can't be drawn as a wedge (start == end) — use a full circle.
  const solo = slices.find((s) => s.value / total >= 0.999);
  let a = -Math.PI / 2; // start at 12 o'clock
  return (
    <svg viewBox={`0 0 ${S} ${S}`} width={S} height={S} className="shrink-0" role="img" aria-label="action probability pie">
      {solo ? (
        <circle cx={c} cy={c} r={r} fill={solo.color} />
      ) : (
        slices.map((s) => {
          const a0 = a;
          const a1 = a + (s.value / total) * 2 * Math.PI;
          a = a1;
          return <path key={s.label} d={wedge(c, c, r, a0, a1)} fill={s.color} stroke="white" strokeWidth={1} />;
        })
      )}
    </svg>
  );
}

// Live per-person action distribution as a pie (top-3 classes + "other"), synced
// to the playing video's currentTime. One panel per tracked person; the pie is
// the softmax distribution for the window covering the current instant — the
// classes compete for one whole (they sum to 1), so "other" = 1 - sum(top 3).
// Reads the action sidecar directly via /api/saved/file (static once analyzed).
export function ActionBars({ relPath, model, currentTime }: { relPath: string; model: string; currentTime: number }) {
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
    <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
      {ids.map((id) => {
        const at = distAt(tracks[id], currentTime);
        const idle = !at || at.ref.kept === false;
        const sorted = at ? [...at.dist.entries()].sort((a, b) => b[1] - a[1]) : [];

        const head = sorted.slice(0, TOP_N);
        const sumHead = head.reduce((s, [, p]) => s + p, 0);
        const other = Math.max(0, 1 - sumHead);
        const slices: Slice[] = [
          ...head.map(([label, value], i) => ({ label, value, color: COLORS[i % COLORS.length] })),
          ...(other > 0.005 ? [{ label: "other", value: other, color: OTHER_COLOR }] : []),
        ];

        return (
          <div key={id} className="rounded-xl border border-line bg-background p-2">
            <div className="mb-1 flex items-center justify-between text-xs font-bold">
              <span>#{id}</span>
              <span className={idle ? "text-muted" : "text-emerald-600"}>{idle ? "idle" : at?.ref.action}</span>
            </div>
            {slices.length === 0 ? (
              <div className="text-[10px] text-muted">—</div>
            ) : (
              <div className="flex items-center gap-3">
                <Pie slices={slices} />
                <div className="flex flex-1 flex-col gap-0.5">
                  {slices.map((s) => (
                    <div key={s.label} className="flex items-center gap-1.5 text-[10px]">
                      <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: s.color }} />
                      <span className="flex-1 truncate text-muted" title={s.label}>{s.label}</span>
                      <span className="shrink-0 font-mono text-muted">{Math.round(s.value * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
