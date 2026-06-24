"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";

import { DATASETS } from "@/lib/action-classes";

// Browse the full label set each action model can emit. The lists are
// index-aligned with the model heads (the number shown is the class index).
export function ActionClassesPage() {
  const [q, setQ] = useState("");
  const query = q.trim().toLowerCase();

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-extrabold">Action classes</h2>
          <p className="text-sm text-muted">
            Every label each model can predict — these are the only actions it can ever output.
          </p>
        </div>
        <label className="flex items-center gap-2 rounded-xl border border-line bg-card px-3 py-1.5">
          <Search className="size-4 text-muted" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="filter classes…"
            className="w-40 bg-transparent text-sm outline-none placeholder:text-muted"
          />
        </label>
      </div>

      <div className="flex flex-col gap-6">
        {DATASETS.map((ds) => (
          <DatasetCard key={ds.key} ds={ds} query={query} />
        ))}
      </div>
    </div>
  );
}

function DatasetCard({ ds, query }: { ds: (typeof DATASETS)[number]; query: string }) {
  // Keep original indices so the chip numbers stay aligned with the model head.
  const rows = useMemo(
    () =>
      ds.classes
        .map((name, i) => ({ name, i }))
        .filter(({ name }) => !query || name.toLowerCase().includes(query)),
    [ds.classes, query],
  );

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-base font-extrabold">{ds.label}</div>
          <div className="text-xs text-muted">
            <span className="font-mono">{ds.model}</span> · {ds.dataset}
          </div>
        </div>
        <span className="rounded-full bg-background px-2.5 py-1 text-xs font-bold text-muted">
          {query ? `${rows.length} / ${ds.classes.length}` : `${ds.classes.length} classes`}
        </span>
      </div>
      <p className="mb-3 text-sm text-muted">{ds.blurb}</p>

      {rows.length === 0 ? (
        <div className="text-sm text-muted">No classes match “{query}”.</div>
      ) : (
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-4">
          {rows.map(({ name, i }) => (
            <div
              key={i}
              className="flex items-center gap-2 rounded-lg border border-line bg-background px-2 py-1.5 text-sm"
            >
              <span className="w-6 shrink-0 text-right font-mono text-[11px] text-muted">{i}</span>
              <span className="truncate" title={name}>
                {name}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
