import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { savedRoot } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Last-run stats per analysis kind, written by detect.py / action.py as
// .last_run.<kind>.json in the recordings root. Newest first.
export async function GET() {
  const root = savedRoot();
  let runs: unknown[] = [];
  try {
    const names = (await readdir(root)).filter(
      (n) => n.startsWith(".last_run.") && n.endsWith(".json"),
    );
    runs = (
      await Promise.all(
        names.map(async (n) => {
          try {
            return JSON.parse(await readFile(path.join(root, n), "utf8"));
          } catch {
            return null;
          }
        }),
      )
    )
      .filter(Boolean)
      .sort((a, b) => String((b as { finishedAt?: string }).finishedAt ?? "").localeCompare(String((a as { finishedAt?: string }).finishedAt ?? "")));
  } catch {
    runs = [];
  }
  return NextResponse.json({ runs }, { headers: { "cache-control": "no-store" } });
}
