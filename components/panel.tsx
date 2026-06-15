"use client";

import { useQuery } from "@tanstack/react-query";

import { NodeCard } from "@/components/node-card";
import { RecordBar } from "@/components/record-bar";
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
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {nodes.map((n) => (
          <NodeCard key={n.id} node={n} />
        ))}
      </div>
      <RecordBar anyRunning={anyRunning} />
    </div>
  );
}
