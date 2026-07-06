"use client";

import { useQuery } from "@tanstack/react-query";

import { RecordBar } from "@/components/record-bar";
import { StreamTile } from "@/components/stream-tile";
import type { CombinedStatus, NodeConfig, NodeStatus } from "@/lib/types";

export function Panel({ nodes: initial }: { nodes: NodeConfig[] }) {
  const { data } = useQuery({
    queryKey: ["status"],
    queryFn: async (): Promise<CombinedStatus> => {
      const res = await fetch("/api/status", { cache: "no-store" });
      return res.json();
    },
    refetchInterval: 1000,
    refetchOnWindowFocus: false,
  });

  // Until the first poll lands, render the configured nodes as offline tiles.
  const nodes: NodeStatus[] =
    data?.nodes ?? initial.map((n) => ({ ...n, online: false, status: null }));
  const anyRunning = nodes.some((n) => n.online && n.status?.running);

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {nodes.map((n) => (
          <StreamTile key={n.id} node={n} />
        ))}
      </div>
      <RecordBar anyRunning={anyRunning} />
    </div>
  );
}
