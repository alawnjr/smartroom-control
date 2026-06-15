"use client";

import { useEffect, useState } from "react";

import type { NodeStatus } from "@/lib/types";

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
}

// One camera tile. The Pi serves /stream.mjpg live, but returns 503 while it's
// recording — so during a recording we swap to polling the recorder's
// /preview.jpg still (~1/s) and show a REC badge, then swap back when it ends.
export function StreamTile({ node }: { node: NodeStatus }) {
  const base = `http://${node.host}:8000`;
  const running = Boolean(node.online && node.status?.running);
  const [streamKey, setStreamKey] = useState(0); // bump to force MJPEG reconnect
  const [tick, setTick] = useState(0); // cache-buster for the preview still
  const [previewOk, setPreviewOk] = useState(false);

  useEffect(() => {
    if (!running) return;
    setTick((t) => t + 1);
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-neutral-800 bg-neutral-900/50 p-3">
      <div className="flex items-center justify-between">
        <span className="font-medium">{node.name}</span>
        <StatusPill online={node.online} running={running} />
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
        {!node.online ? (
          <div className="flex h-full items-center justify-center text-sm text-neutral-500">
            offline / unreachable
          </div>
        ) : running ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${base}/preview.jpg?t=${tick}`}
              alt={`${node.name} recording preview`}
              className="h-full w-full object-contain"
              onLoad={() => setPreviewOk(true)}
              onError={() => setPreviewOk(false)}
            />
            {!previewOk && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-neutral-400">
                starting…
              </div>
            )}
            <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-red-600/90 px-2 py-0.5 text-xs font-semibold text-white">
              <span className="size-2 animate-pulse rounded-full bg-white" /> REC
            </span>
            {node.status && (
              <span className="absolute bottom-2 right-2 rounded-md bg-black/70 px-2 py-0.5 text-xs tabular-nums text-neutral-200">
                {fmt(node.status.remaining)} left
              </span>
            )}
          </>
        ) : (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={streamKey}
              src={`${base}/stream.mjpg`}
              alt={`${node.name} live`}
              className="h-full w-full object-contain"
              onError={() => setTimeout(() => setStreamKey((k) => k + 1), 1500)}
            />
          </>
        )}
      </div>

      <span className="text-xs text-neutral-500">{node.host}</span>
    </div>
  );
}

function StatusPill({ online, running }: { online: boolean; running: boolean }) {
  const [color, label] = !online
    ? ["bg-neutral-600", "offline"]
    : running
      ? ["bg-red-500", "recording"]
      : ["bg-emerald-500", "live"];
  return (
    <span className="flex items-center gap-1.5 text-xs text-neutral-400">
      <span className={`size-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
