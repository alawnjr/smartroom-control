"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { barClass } from "@/lib/action-colors";

// Per-window classifier output recorded by detect/action.py (camera_main.actions.<model>.json).
type Entry = { t: number; action: string; conf: number; kept?: boolean; top?: [string, number][] };
type Timeline = { tracks: Record<string, Entry[]> };

// How many top classes to show as bars, and the confidence a full-width bar maps
// to (these heads rarely exceed ~0.5; near-uniform reads as honestly short bars).
const TOP_N = 5;
const SCALE = 0.6;

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
    <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
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
          </div>
        );
      })}
    </div>
  );
}
