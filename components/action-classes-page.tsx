"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Search } from "lucide-react";

import { DATASETS } from "@/lib/action-classes";

type Config = Record<string, { disabled?: string[]; stride?: number }>;

const STRIDE_OPTS = [0, 1, 2, 3, 4]; // 0 = auto (fps-adaptive)

// Browse the full label set each action model can emit, and toggle classes on/off.
// Disabled classes are masked at inference (detect/action.py) so the model can
// never predict them — useful for killing domain-mismatched labels (HMDB's
// dive/dribble, NTU's medical classes) that misfire on room footage.
export function ActionClassesPage() {
  const [q, setQ] = useState("");
  const query = q.trim().toLowerCase();

  const qc = useQueryClient();
  const { data: config } = useQuery({
    queryKey: ["action-classes"],
    queryFn: async (): Promise<Config> => {
      const res = await fetch("/api/action-classes", { cache: "no-store" });
      return res.ok ? res.json() : {};
    },
    // Load once; the cache is the source of truth between saves (we update it
    // optimistically), so we don't refetch and clobber an in-flight toggle.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const stride = config?.settings?.stride ?? 0;
  const saveStride = useMutation({
    mutationFn: async (n: number) => {
      const res = await fetch("/api/action-classes", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ stride: n }),
      });
      if (!res.ok) throw new Error("save failed");
    },
    onMutate: (n: number) =>
      qc.setQueryData<Config>(["action-classes"], (old) => ({
        ...(old ?? {}),
        settings: { ...old?.settings, stride: n },
      })),
    onError: () => qc.invalidateQueries({ queryKey: ["action-classes"] }),
  });

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-extrabold">Action classes</h2>
          <p className="text-sm text-muted">
            Toggle classes off to mask them at inference — the model picks the best{" "}
            <em>enabled</em> class or falls back to idle. Re-analyze clips to apply.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2" title="Frames between samples in the classifier window. Auto adapts to each clip's true fps to target a ~3.2s window.">
            <span className="text-xs font-bold text-muted">stride</span>
            <div className="flex overflow-hidden rounded-lg border border-line text-xs font-bold">
              {STRIDE_OPTS.map((n) => (
                <button
                  key={n}
                  onClick={() => saveStride.mutate(n)}
                  className={`px-2 py-1 ${n === stride ? "bg-emerald-500 text-white" : "text-muted hover:bg-background"}`}
                >
                  {n === 0 ? "auto" : n}
                </button>
              ))}
            </div>
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
      </div>

      <div className="flex flex-col gap-6">
        {DATASETS.map((ds) => (
          <DatasetCard key={ds.key} ds={ds} query={query} disabled={config?.[ds.key]?.disabled ?? []} />
        ))}
      </div>
    </div>
  );
}

function DatasetCard({
  ds,
  query,
  disabled,
}: {
  ds: (typeof DATASETS)[number];
  query: string;
  disabled: string[];
}) {
  const qc = useQueryClient();
  const disabledSet = useMemo(() => new Set(disabled), [disabled]);

  // Persist the new disabled list for this variant, optimistically updating cache.
  // We do NOT refetch on success — the optimistic cache is authoritative and a
  // refetch could revert an in-flight toggle. On error we roll back to the server.
  const save = useMutation({
    mutationFn: async (next: string[]) => {
      const res = await fetch("/api/action-classes", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ variant: ds.key, disabled: next }),
      });
      if (!res.ok) throw new Error(`save failed (${res.status})`);
      return next;
    },
    onMutate: (next: string[]) => {
      qc.setQueryData<Config>(["action-classes"], (old) => ({
        ...(old ?? {}),
        [ds.key]: { disabled: next },
      }));
    },
    onError: () => qc.invalidateQueries({ queryKey: ["action-classes"] }),
  });

  // Read the freshest disabled list from the cache at click time (not the
  // render-time prop) so rapid toggles don't overwrite each other.
  const current = () =>
    new Set(qc.getQueryData<Config>(["action-classes"])?.[ds.key]?.disabled ?? disabled);
  const toggle = (name: string) => {
    const next = current();
    if (next.has(name)) next.delete(name);
    else next.add(name);
    save.mutate([...next]);
  };
  const setAll = (off: boolean) => save.mutate(off ? [...ds.classes] : []);

  // Toggles only take effect when clips are re-analyzed (the mask is baked in at
  // analysis time). Force a re-run of this variant so the change actually applies.
  const variant = ds.key === "action-hmdb" ? "hmdb" : "ntu";
  const reanalyze = useMutation({
    mutationFn: () =>
      fetch("/api/action", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ force: true, variant }),
      }),
  });

  // Keep original indices so the chip numbers stay aligned with the model head.
  const rows = useMemo(
    () =>
      ds.classes
        .map((name, i) => ({ name, i }))
        .filter(({ name }) => !query || name.toLowerCase().includes(query)),
    [ds.classes, query],
  );
  const enabledCount = ds.classes.length - disabledSet.size;

  return (
    <div className="overflow-hidden rounded-[22px] border border-line bg-card p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-base font-extrabold">{ds.label}</div>
          <div className="text-xs text-muted">
            <span className="font-mono">{ds.model}</span> · {ds.dataset}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-background px-2.5 py-1 text-xs font-bold text-muted">
            {enabledCount} / {ds.classes.length} on
          </span>
          <button
            onClick={() => setAll(false)}
            className="rounded-lg border border-line px-2 py-1 text-[11px] font-bold text-muted hover:bg-background"
          >
            all on
          </button>
          <button
            onClick={() => setAll(true)}
            className="rounded-lg border border-line px-2 py-1 text-[11px] font-bold text-muted hover:bg-background"
          >
            all off
          </button>
          <button
            onClick={() => reanalyze.mutate()}
            disabled={reanalyze.isPending}
            title="Re-run this model on all clips so the toggles take effect"
            className="flex items-center gap-1 rounded-lg bg-emerald-500 px-2 py-1 text-[11px] font-bold text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            <RefreshCw className={`size-3 ${reanalyze.isPending ? "animate-spin" : ""}`} />
            {reanalyze.isSuccess ? "re-analyzing…" : "apply"}
          </button>
        </div>
      </div>
      <p className="mb-3 text-sm text-muted">{ds.blurb}</p>

      {rows.length === 0 ? (
        <div className="text-sm text-muted">No classes match “{query}”.</div>
      ) : (
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-4">
          {rows.map(({ name, i }) => {
            const on = !disabledSet.has(name);
            return (
              <button
                key={i}
                onClick={() => toggle(name)}
                title={on ? "enabled — click to disable" : "disabled — click to enable"}
                className={`flex items-center gap-2 rounded-lg border px-2 py-1.5 text-left text-sm transition-colors ${
                  on
                    ? "border-line bg-background hover:bg-card"
                    : "border-dashed border-line bg-transparent text-muted line-through opacity-60"
                }`}
              >
                <span
                  className={`flex h-3.5 w-6 shrink-0 items-center rounded-full px-0.5 transition-colors ${
                    on ? "justify-end bg-emerald-400" : "justify-start bg-neutral-300"
                  }`}
                >
                  <span className="size-2.5 rounded-full bg-white" />
                </span>
                <span className="w-5 shrink-0 text-right font-mono text-[11px] text-muted">{i}</span>
                <span className="truncate" title={name}>
                  {name}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
