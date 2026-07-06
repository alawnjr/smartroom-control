"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

// Per-window classifier output recorded by detect/action.py (camera_main.actions.<model>.json).
type Entry = { t: number; action: string; conf: number; kept?: boolean; top?: [string, number][] };
type Timeline = { tracks: Record<string, Entry[]> };

// Bars are scaled against this absolute confidence so their length reflects the
// real magnitude (these heads rarely exceed ~0.5; near-uniform = honestly short).
const SCALE = 0.6;

function actionsPath(relPath: string, model: string) {
  return relPath.replace(/\.mp4$/, `.actions.${model}.json`);
}

// Latest window at or before the playhead (entries are time-ascending).
function latestAt(entries: Entry[], t: number): Entry | null {
  let found: Entry | null = null;
  for (const e of entries) {
    if (e.t <= t) found = e;
    else break;
  }
  return found;
}

// Live per-person "most confident classes" bar graph, synced to the playing
// video's currentTime. One panel per tracked person; bars are the top-K classes
// for the window covering the current instant. Reads the action sidecar directly
// via /api/saved/file (cached — it's static once analyzed).
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
        const cur = latestAt(tracks[id], currentTime);
        const bars: [string, number][] = cur?.top ?? (cur ? [[cur.action, cur.conf]] : []);
        const idle = !cur || cur.kept === false;
        return (
          <div key={id} className="rounded-xl border border-line bg-background p-2">
            <div className="mb-1 flex items-center justify-between text-xs font-bold">
              <span>#{id}</span>
              <span className={idle ? "text-muted" : "text-emerald-600"}>{idle ? "idle" : cur?.action}</span>
            </div>
            <div className="flex flex-col gap-1">
              {bars.length === 0 ? (
                <div className="text-[10px] text-muted">—</div>
              ) : (
                bars.map(([label, p], i) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <span className="w-24 shrink-0 truncate text-[10px] text-muted" title={label}>
                      {label}
                    </span>
                    <div className="relative h-2.5 flex-1 overflow-hidden rounded bg-line/40">
                      <div
                        className={`absolute inset-y-0 left-0 rounded ${i === 0 && !idle ? "bg-emerald-500" : "bg-emerald-300"}`}
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
