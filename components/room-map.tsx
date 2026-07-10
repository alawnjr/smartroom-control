"use client";

import { useQuery } from "@tanstack/react-query";
import { MapPin, Video } from "lucide-react";

import { groupSessions, useSaved } from "@/lib/use-saved";
import type { SavedVideo } from "@/lib/types";

// Top-down plan view of each located clip: the AprilTag's wall along the top,
// per-track walking paths in metric room coordinates. Data comes from the
// centroids sidecar written by detect/action.py — entries carry room:[x,z] mm
// (relative to the floor point under the tag) when the clip has intrinsic +
// extrinsic calibration and a solved tag height (detect/floor_calib.py).

type RoomFrame = {
  tagId?: number;
  tagHeightMm?: number;
  cameraPositionMm?: [number, number, number];
};
type CentEntry = { t: number; room?: [number, number] | null; src?: string };
type CentroidsFile = {
  roomFrame?: RoomFrame;
  persons?: Record<string, CentEntry[]>;
};

function fileUrl(relPath: string) {
  return `/api/saved/file?path=${encodeURIComponent(relPath)}`;
}
function centroidsPath(relPath: string, model: string) {
  return relPath.replace(/\.mp4$/, `.centroids.${model}.json`);
}

const TRACK_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];

function ClipMap({ video, model, version }: { video: SavedVideo; model: string; version?: number }) {
  const rel = centroidsPath(video.relPath, model);
  const q = useQuery<CentroidsFile>({
    queryKey: ["centroids", rel, version ?? 0],
    queryFn: () => fetch(fileUrl(rel)).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    staleTime: Infinity,
    retry: false,
  });
  const data = q.data;
  if (!data?.roomFrame) return null; // clip isn't located (or sidecar predates room positions)

  const cam = data.roomFrame.cameraPositionMm ?? [0, 0, 0];
  const tracks = Object.entries(data.persons ?? {})
    .map(([tid, entries]) => ({
      tid,
      pts: entries.filter((e): e is CentEntry & { room: [number, number] } => Array.isArray(e.room)),
    }))
    .filter((tr) => tr.pts.length >= 2);
  if (tracks.length === 0) return null;

  // Bounds over everything drawn (mm): paths + tag origin + camera, padded.
  const xs = [0, cam[0]], zs = [0, cam[2]];
  for (const tr of tracks) for (const p of tr.pts) { xs.push(p.room[0]); zs.push(p.room[1]); }
  const pad = 400;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minZ = Math.min(...zs) - pad, maxZ = Math.max(...zs) + pad;
  const W = 340, H = Math.max(220, Math.min(420, (W * (maxZ - minZ)) / (maxX - minX)));
  // Plan view: X right; the wall (Z=0) at the top, the room grows downward.
  const sx = (x: number) => ((x - minX) / (maxX - minX)) * W;
  const sy = (z: number) => ((z - minZ) / (maxZ - minZ)) * H;

  const gridLines = [];
  for (let g = Math.ceil(minX / 1000) * 1000; g <= maxX; g += 1000)
    gridLines.push(<line key={`vx${g}`} x1={sx(g)} y1={0} x2={sx(g)} y2={H} className="stroke-line" strokeWidth={0.5} />);
  for (let g = Math.ceil(minZ / 1000) * 1000; g <= maxZ; g += 1000)
    gridLines.push(<line key={`hz${g}`} x1={0} y1={sy(g)} x2={W} y2={sy(g)} className="stroke-line" strokeWidth={0.5} />);

  return (
    <div className="rounded-xl border border-line bg-card p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-bold">
          {video.node} · {video.rec}
        </span>
        <span className="text-xs text-muted">
          tag {data.roomFrame.tagId ?? "?"} at origin · grid 1 m
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-line">
        {gridLines}
        {/* the tag's wall (Z = 0) */}
        <line x1={0} y1={sy(0)} x2={W} y2={sy(0)} className="stroke-muted" strokeWidth={1.5} />
        {/* tag marker at the origin */}
        <rect x={sx(0) - 5} y={sy(0) - 3} width={10} height={6} fill="#e11d48" rx={1} />
        {/* camera marker */}
        <g transform={`translate(${sx(cam[0])}, ${sy(cam[2])})`}>
          <circle r={5} fill="#0ea5e9" />
          <line x1={0} y1={0} x2={(sx(0) - sx(cam[0])) * 0.25} y2={(sy(0) - sy(cam[2])) * 0.25}
                stroke="#0ea5e9" strokeWidth={1.5} />
        </g>
        {tracks.map((tr, i) => {
          const color = TRACK_COLORS[i % TRACK_COLORS.length];
          const d = tr.pts.map((p, j) => `${j ? "L" : "M"}${sx(p.room[0]).toFixed(1)},${sy(p.room[1]).toFixed(1)}`).join("");
          const last = tr.pts[tr.pts.length - 1];
          return (
            <g key={tr.tid}>
              <path d={d} fill="none" stroke={color} strokeWidth={1.6} strokeOpacity={0.75}
                    strokeLinejoin="round" strokeLinecap="round" />
              <circle cx={sx(tr.pts[0].room[0])} cy={sy(tr.pts[0].room[1])} r={2.5} fill={color} fillOpacity={0.5} />
              <circle cx={sx(last.room[0])} cy={sy(last.room[1])} r={3.5} fill={color} />
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-[2px] bg-[#e11d48]" /> tag</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full bg-[#0ea5e9]" /> camera</span>
        {tracks.map((tr, i) => (
          <span key={tr.tid} className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: TRACK_COLORS[i % TRACK_COLORS.length] }} />
            person {tr.tid} ({tr.pts.length} pts)
          </span>
        ))}
      </div>
    </div>
  );
}

export function RoomMapPanel() {
  const saved = useSaved();
  const videos = saved.data?.videos ?? [];

  // Clips with a finished action analysis (whose centroids sidecar may carry
  // room positions; ClipMap renders nothing for the ones that don't).
  const candidates = videos
    .map((v) => {
      const hit = Object.entries(v.detections ?? {}).find(
        ([m, d]) => m.startsWith("action") && d.status === "done",
      );
      return hit ? { v, model: hit[0], version: hit[1].version } : null;
    })
    .filter(Boolean) as { v: SavedVideo; model: string; version?: number }[];

  const sessions = groupSessions(videos)
    .map((s) => ({ ...s, clips: s.clips.filter((c) => candidates.some((x) => x.v.relPath === c.relPath)) }))
    .filter((s) => s.clips.length > 0);
  const byClip = new Map(candidates.map((x) => [x.v.relPath, x]));

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-lg font-extrabold flex items-center gap-2">
          <MapPin size={18} /> Room map
        </h2>
        <p className="text-sm text-muted">
          Top-down metric view of where each person <strong>stood on the floor</strong>, in the
          AprilTag&apos;s coordinate frame (X along the wall, Z out from it) — computed from ankle
          keypoints + the camera&apos;s tag calibration. Only clips recorded with intrinsic +
          extrinsic calibration appear; re-analyze a clip to populate it.
        </p>
      </div>
      {sessions.length === 0 ? (
        <p className="text-sm text-muted flex items-center gap-2">
          <Video size={14} /> No located clips yet — calibrate the camera (intrinsic + extrinsic),
          run the floor solve, then re-analyze.
        </p>
      ) : (
        <div className="flex flex-col gap-7">
          {sessions.map((s) => (
            <div key={s.key}>
              <h3 className="mb-2 text-sm font-bold text-muted">{s.label}</h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {s.clips.map((c) => {
                  const x = byClip.get(c.relPath)!;
                  return <ClipMap key={c.relPath} video={c} model={x.model} version={x.version} />;
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
