"use client";

import type { NodeStatus } from "@/lib/types";

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
}

// Compact per-node status card (no live video). Shows online/recording state and
// a progress bar + countdown while recording.
export function NodeCard({ node }: { node: NodeStatus }) {
  const running = Boolean(node.online && node.status?.running);
  const [color, label] = !node.online
    ? ["bg-neutral-600", "offline"]
    : running
      ? ["bg-red-500", "recording"]
      : ["bg-emerald-500", "idle"];

  const pct =
    running && node.status?.duration
      ? Math.min(100, (node.status.elapsed / node.status.duration) * 100)
      : 0;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="font-medium">{node.name}</span>
          <span className="text-xs text-neutral-500">{node.host}</span>
        </div>
        <span className="flex items-center gap-1.5 text-xs text-neutral-400">
          <span className={`size-2 rounded-full ${color} ${running ? "animate-pulse" : ""}`} />
          {label}
        </span>
      </div>

      {running && node.status && (
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between text-xs tabular-nums text-neutral-400">
            <span>{fmt(node.status.elapsed)} elapsed</span>
            <span>{fmt(node.status.remaining)} left</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-800">
            <div
              className="h-full bg-red-500 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
