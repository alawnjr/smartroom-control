"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { RecordResponse } from "@/lib/types";

async function post(url: string, body?: unknown): Promise<RecordResponse> {
  const res = await fetch(url, {
    method: "POST",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

export function RecordBar({ anyRunning }: { anyRunning: boolean }) {
  const [duration, setDuration] = useState(30);
  const [results, setResults] = useState<RecordResponse["results"]>([]);
  const qc = useQueryClient();

  const settle = (r: RecordResponse) => {
    setResults(r.results);
    qc.invalidateQueries({ queryKey: ["status"] }); // flip tiles immediately
  };
  const record = useMutation({
    mutationFn: () => post("/api/record", { duration }),
    onSuccess: settle,
  });
  const cancel = useMutation({ mutationFn: () => post("/api/cancel"), onSuccess: settle });
  const busy = record.isPending || cancel.isPending;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-sm text-neutral-400">
          Duration (seconds)
          <Input
            type="number"
            min={1}
            max={3600}
            value={duration}
            onChange={(e) =>
              setDuration(Math.max(1, Math.min(3600, Number(e.target.value) || 1)))
            }
            className="w-32"
          />
        </label>
        <Button size="lg" disabled={busy || anyRunning} onClick={() => record.mutate()}>
          {record.isPending ? "Starting…" : "Record All"}
        </Button>
        <Button
          size="lg"
          variant="destructive"
          disabled={busy || !anyRunning}
          onClick={() => cancel.mutate()}
        >
          {cancel.isPending ? "Cancelling…" : "Cancel All"}
        </Button>
      </div>

      {results.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-xs">
          {results.map((r) => (
            <li key={r.id} className={r.ok ? "text-neutral-400" : "text-red-400"}>
              {r.name}: {r.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
