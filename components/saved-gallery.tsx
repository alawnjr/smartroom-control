"use client";

import { useQuery } from "@tanstack/react-query";

import type { SavedListing, SavedVideo } from "@/lib/types";

function mb(b: number) {
  return `${(b / 1e6).toFixed(b >= 1e7 ? 0 : 1)} MB`;
}

export function SavedGallery() {
  const { data } = useQuery({
    queryKey: ["saved"],
    queryFn: async (): Promise<SavedListing> => {
      const res = await fetch("/api/saved", { cache: "no-store" });
      return res.json();
    },
    refetchOnWindowFocus: false,
  });

  const videos = data?.videos ?? [];
  const byNode = new Map<string, SavedVideo[]>();
  for (const v of videos) {
    const arr = byNode.get(v.node);
    if (arr) arr.push(v);
    else byNode.set(v.node, [v]);
  }
  const nodes = [...byNode.keys()].sort();

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
      <h2 className="mb-3 text-sm font-medium text-neutral-300">Saved Recordings</h2>
      {videos.length === 0 ? (
        <p className="text-xs text-neutral-500">
          No saved recordings yet — hit “Save All to Laptop”.
        </p>
      ) : (
        <div className="flex flex-col gap-5">
          {nodes.map((node) => (
            <div key={node} className="flex flex-col gap-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
                {node}{" "}
                <span className="font-normal lowercase text-neutral-600">
                  ({byNode.get(node)!.length})
                </span>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {byNode.get(node)!.map((v) => {
                  const url = `/api/saved/file?path=${encodeURIComponent(v.relPath)}`;
                  return (
                    <div
                      key={v.relPath}
                      className="flex flex-col gap-1 rounded-lg border border-neutral-800 bg-black/30 p-2"
                    >
                      <video
                        controls
                        preload="none"
                        className="aspect-video w-full rounded bg-black"
                        src={url}
                      />
                      <div className="flex items-center justify-between gap-2 text-xs text-neutral-400">
                        <span className="truncate" title={`${v.day}/${v.rec}/${v.file}`}>
                          {v.rec || v.file}
                        </span>
                        <a
                          className="shrink-0 text-emerald-400 hover:underline"
                          href={url}
                          download={`${node}_${v.rec}_${v.file}`}
                        >
                          download
                        </a>
                      </div>
                      <span className="text-[10px] text-neutral-600">
                        {v.day} · {mb(v.size)}
                      </span>
                    </div>
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
