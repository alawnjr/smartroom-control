import { useQuery } from "@tanstack/react-query";

import type { SavedListing, SavedVideo } from "@/lib/types";

// True while any model on a clip is still being analyzed.
export function clipAnalyzing(v: SavedVideo) {
  return Object.values(v.detections ?? {}).some((d) => d.status === "analyzing");
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
