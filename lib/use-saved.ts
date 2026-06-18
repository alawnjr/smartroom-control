import { useQuery, type QueryClient } from "@tanstack/react-query";

import type { SavedListing, SavedVideo } from "@/lib/types";

// After triggering/cancelling detection, re-check ["saved"] a few times over the
// next several seconds. detect.py needs a moment to write its first "analyzing"
// marker, so a single immediate invalidate would miss it and polling (which only
// runs while something is analyzing) would never start. Once a ping catches the
// analyzing state, useSaved's refetchInterval self-sustains.
export function pingSavedSoon(qc: QueryClient) {
  for (const ms of [300, 1000, 2000, 4000, 7000, 11000]) {
    setTimeout(() => qc.invalidateQueries({ queryKey: ["saved"] }), ms);
  }
}

// True while any model on a clip is still being analyzed.
export function clipAnalyzing(v: SavedVideo) {
  return Object.values(v.detections ?? {}).some((d) => d.status === "analyzing");
}

// A recording "session" = clips captured together (same day/rec across cameras,
// which is what a single "Record everything" produces). Newest first.
export type Session = { key: string; label: string; clips: SavedVideo[]; mtime: number };

export function groupSessions(videos: SavedVideo[]): Session[] {
  const map = new Map<string, SavedVideo[]>();
  for (const v of videos) {
    const key = `${v.day}/${v.rec}`;
    const arr = map.get(key);
    if (arr) arr.push(v);
    else map.set(key, [v]);
  }
  const sessions = [...map.entries()].map(([key, clips]) => {
    const { day, rec } = clips[0];
    const date = day.replace(/^day_\d+_/, ""); // YYYY-MM-DD
    const take = rec.split("_").pop() ?? rec;
    return {
      key,
      label: `${date} · take ${take}`,
      clips: [...clips].sort((a, b) => a.node.localeCompare(b.node)),
      mtime: Math.max(...clips.map((c) => c.mtime)),
    };
  });
  sessions.sort((a, b) => b.mtime - a.mtime);
  return sessions;
}

export function analyzingCount(listing?: SavedListing) {
  return listing?.videos?.filter(clipAnalyzing).length ?? 0;
}

// Shared ["saved"] query — both the gallery and the save/analyze bar read it (React
// Query dedupes to one fetch). Polls every 2s while anything is analyzing so the UI
// reflects progress live.
export function useSaved() {
  return useQuery({
    queryKey: ["saved"],
    queryFn: async (): Promise<SavedListing> => {
      const res = await fetch("/api/saved", { cache: "no-store" });
      return res.json();
    },
    refetchOnWindowFocus: false,
    refetchInterval: (q) => (analyzingCount(q.state.data) > 0 ? 2000 : false),
  });
}
