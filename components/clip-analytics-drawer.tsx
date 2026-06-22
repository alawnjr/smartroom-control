"use client";

import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

import { ActionBars } from "@/components/action-bars";
import { tagClass } from "@/lib/action-colors";
import type { DetectionSummary, SavedVideo } from "@/lib/types";

const SPEEDS = [0.25, 0.5, 1, 2];

function fileUrl(relPath: string) {
  return `/api/saved/file?path=${encodeURIComponent(relPath)}`;
}

// Right slide-over showing one clip's per-person action analysis: the video (with
// speed + raw/labels toggle) driving the live bars and the probability-over-time
// line graph. Self-contained — mounted only while open, so the video and the
// per-frame rAF sampling don't run until the user opens it.
export function ClipAnalyticsDrawer({
  v,
  model,
  d,
  roomName,
  onClose,
}: {
  v: SavedVideo;
  model: string;
  d: DetectionSummary;
  roomName: string;
  onClose: () => void;
}) {
  const hasOverlay = Boolean(d.hasAnnotated && d.annotatedRelPath);
  const [overlay, setOverlay] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [rate, setRate] = useState(1);
  const videoRef = useRef<HTMLVideoElement>(null);
  const showOverlay = overlay && hasOverlay;
  const src = showOverlay ? fileUrl(d.annotatedRelPath!) : fileUrl(v.relPath);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Sample currentTime every animation frame while playing so the bars/lines
  // glide between windows (onTimeUpdate alone fires ~4x/sec).
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    let raf = 0;
    const tick = () => {
      setCurrentTime(el.currentTime);
      raf = requestAnimationFrame(tick);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      setCurrentTime(el.currentTime);
    };
    el.addEventListener("playing", start);
    el.addEventListener("pause", stop);
    el.addEventListener("ended", stop);
    el.addEventListener("seeked", stop);
    if (!el.paused) start();
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("playing", start);
      el.removeEventListener("pause", stop);
      el.removeEventListener("ended", stop);
      el.removeEventListener("seeked", stop);
    };
  }, [src]);

  // Apply playback speed (a fresh <video> on src change resets to 1x).
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate, src]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative h-full w-[460px] max-w-[92vw] overflow-y-auto bg-card p-4 shadow-2xl">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-sm font-extrabold">
            {roomName} <span className="font-mono text-xs font-normal text-muted">· {v.rec.split("_").pop()}</span>
          </div>
          <button onClick={onClose} title="Close" className="rounded-lg border border-line p-1.5 text-muted hover:bg-background">
            <X className="size-4" />
          </button>
        </div>

        <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <video
            key={src}
            ref={videoRef}
            controls
            autoPlay
            preload="auto"
            className="h-full w-full object-contain"
            src={src}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          />
          {hasOverlay && (
            <button
              onClick={() => setOverlay((o) => !o)}
              className="absolute bottom-2 right-2 rounded-md border border-white/30 bg-black/60 px-2 py-0.5 text-[10px] font-bold text-white"
            >
              {showOverlay ? "raw" : "labels"}
            </button>
          )}
        </div>

        <div className="mt-2 flex items-center gap-2">
          <span className="text-[10px] font-bold text-muted">speed</span>
          <div className="flex overflow-hidden rounded-lg border border-line text-[10px] font-bold">
            {SPEEDS.map((r) => (
              <button
                key={r}
                onClick={() => setRate(r)}
                className={`px-1.5 py-1 ${r === rate ? "bg-emerald-500 text-white" : "text-muted hover:bg-background"}`}
              >
                {r}×
              </button>
            ))}
          </div>
        </div>

        {(d.actions ?? []).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {(d.actions ?? []).slice(0, 8).map((t, i) => (
              <span key={t} className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${tagClass(i)}`}>{t}</span>
            ))}
          </div>
        )}

        <ActionBars relPath={v.relPath} model={model} currentTime={currentTime} actions={d.actions ?? []} />
      </div>
    </div>
  );
}
