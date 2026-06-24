"use client";

import { useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, PersonStanding } from "lucide-react";

import { groupSessions, useSaved } from "@/lib/use-saved";
import type { DetectionSummary, NodeConfig, SavedVideo } from "@/lib/types";

type JumpEv = { start: number; end: number; peak: number };
type ActionsFile = { jumps?: Record<string, JumpEv[]> };

function fileUrl(relPath: string) {
  return `/api/saved/file?path=${encodeURIComponent(relPath)}`;
}
function actionsPath(relPath: string, model: string) {
  return relPath.replace(/\.mp4$/, `.actions.${model}.json`);
}

// Geometric, classifier-independent motion events (currently jumps) detected from
// the pose trajectory in detect/action.py. One card per clip that has any event.
export function GeometricPage({ nodes: config }: { nodes: NodeConfig[] }) {
  const saved = useSaved();
  const videos = saved.data?.videos ?? [];
  const nameByNode = new Map(config.map((n) => [n.id, n.name]));

  // For each clip, the action model whose analysis found jumps (jumps are the same
  // across variants, so just take the first that has them).
  const withJumps = videos
    .map((v) => {
      const hit = Object.entries(v.detections ?? {}).find(
        ([m, d]) => m.startsWith("action") && d.status === "done" && (d.jumps ?? 0) > 0,
      );
      return hit ? { v, model: hit[0], d: hit[1] } : null;
    })
    .filter(Boolean) as { v: SavedVideo; model: string; d: DetectionSummary }[];

  const byClip = new Map(withJumps.map((x) => [x.v.relPath, x]));
  const sessions = groupSessions(videos)
    .map((s) => ({ ...s, clips: s.clips.filter((c) => byClip.has(c.relPath)) }))
    .filter((s) => s.clips.length > 0);

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-lg font-extrabold">Geometric events</h2>
        <p className="text-sm text-muted">
          Motion events read directly from each person’s skeleton trajectory (physics, not the
          action classifier) — currently <strong>jumps</strong>: the center of mass briefly rising
          above its standing baseline. Re-analyze a clip to populate this.
        </p>
      </div>

      {sessions.length === 0 ? (
        <p className="text-sm text-muted">
          No jumps detected in any analyzed clip yet. Run Actions on a clip that contains a jump.
        </p>
      ) : (
        <div className="flex flex-col gap-7">
          {sessions.map((s) => (
            <div key={s.key}>
              <div className="mb-2 flex items-center gap-2 text-sm font-bold text-muted">
                <span className="font-mono">{s.label}</span>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {s.clips.map((c) => {
                  const x = byClip.get(c.relPath)!;
                  return (
                    <JumpCard
                      key={c.relPath}
                      v={x.v}
                      model={x.model}
                      d={x.d}
                      roomName={nameByNode.get(c.node) ?? c.node}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function JumpCard({
  v,
  model,
  d,
  roomName,
}: {
  v: SavedVideo;
  model: string;
  d: DetectionSummary;
  roomName: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const src = d.hasAnnotated && d.annotatedRelPath ? fileUrl(d.annotatedRelPath) : fileUrl(v.relPath);

  const { data } = useQuery({
    queryKey: ["jumps", v.relPath, model],
    queryFn: async (): Promise<ActionsFile | null> => {
      const res = await fetch(`/api/saved/file?path=${encodeURIComponent(actionsPath(v.relPath, model))}`, {
        cache: "force-cache",
      });
      return res.ok ? res.json().catch(() => null) : null;
    },
    staleTime: 5 * 60_000,
  });

  // Flatten per-track events into one time-sorted list.
  const events = Object.entries(data?.jumps ?? {})
    .flatMap(([id, evs]) => evs.map((e) => ({ id, ...e })))
    .sort((a, b) => a.start - b.start);

  const seek = (t: number) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, t - 0.4); // a beat before the takeoff
    el.play().catch(() => {});
  };

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-extrabold">
          {roomName} <span className="font-mono text-xs font-normal text-muted">· {v.rec.split("_").pop()}</span>
        </div>
        <span className="flex items-center gap-1 rounded-md bg-amber-200 px-1.5 py-0.5 text-[10px] font-bold text-amber-900">
          <ArrowUp className="size-3" /> {d.jumps} jump{d.jumps === 1 ? "" : "s"}
        </span>
      </div>

      <div className="aspect-video w-full overflow-hidden rounded-xl bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <video ref={videoRef} controls preload="metadata" className="h-full w-full object-contain" src={src} />
      </div>

      <div className="mt-2 flex flex-col gap-1">
        {events.length === 0 ? (
          <div className="text-[11px] text-muted">loading events…</div>
        ) : (
          events.map((e, i) => (
            <button
              key={i}
              onClick={() => seek(e.start)}
              title="Jump to this moment"
              className="flex items-center justify-between rounded-lg border border-line bg-background px-2 py-1.5 text-left text-xs hover:bg-card"
            >
              <span className="flex items-center gap-1.5 font-bold">
                <PersonStanding className="size-3.5 text-muted" /> #{e.id}
              </span>
              <span className="font-mono text-[11px] text-muted">
                {e.start.toFixed(1)}–{e.end.toFixed(1)}s · ↑{e.peak.toFixed(2)}
              </span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
