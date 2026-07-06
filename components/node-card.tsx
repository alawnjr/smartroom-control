"use client";

import { useState } from "react";

import type { NodeStatus } from "@/lib/types";

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
}

// Per-node card: live MJPEG preview while idle, a "recording in progress"
// placeholder while recording (the Pi's stream is unavailable then anyway),
// and an offline state. Plus a status pill and a countdown/progress bar.
export function NodeCard({ node }: { node: NodeStatus }) {
  const base = `http://${node.host}:8000`;
  const running = Boolean(node.online && node.status?.running);
  const [streamKey, setStreamKey] = useState(0); // bump to force MJPEG reconnect

  const [color, label] = !node.online
    ? ["bg-neutral-600", "offline"]
    : running
      ? ["bg-red-500", "recording"]
      : ["bg-emerald-500", "live"];

  const pct =
    running && node.status?.duration
      ? Math.min(100, (node.status.elapsed / node.status.duration) * 100)
      : 0;

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-neutral-800 bg-neutral-900/50 p-3">
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

      <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
        {!node.online ? (
          <div className="flex h-full items-center justify-center text-sm text-neutral-500">
            offline / unreachable
          </div>
        ) : running ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-neutral-400">
            <span className="flex items-center gap-1.5 rounded-md bg-red-600/90 px-2 py-0.5 text-xs font-semibold text-white">
              <span className="size-2 animate-pulse rounded-full bg-white" /> REC
            </span>
            <span className="text-sm">Recording in progress…</span>
          </div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={streamKey}
            // cache-buster so an error-retry is a genuinely fresh request (a
            // remount with an identical URL can stick on Chrome's cached error)
            src={`${base}/stream.mjpg?k=${streamKey}`}
            alt={`${node.name} live`}
            className="h-full w-full object-contain"
            onError={() => setTimeout(() => setStreamKey((k) => k + 1), 1500)}
          />
        )}
      </div>

      {running && node.status && (
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between text-xs tabular-nums text-neutral-400">
            <span>{fmt(node.status.elapsed)} elapsed</span>
            <span>{fmt(node.status.remaining)} left</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-800">
            <div className="h-full bg-red-500 transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
