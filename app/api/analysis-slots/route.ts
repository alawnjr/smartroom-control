import { mkdirSync, readdirSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

import { safeResolve } from "@/lib/recordings";
import { spawnBatch } from "@/lib/spawn-batch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function actionPython() {
  return process.env.SMARTROOM_ACTION_PYTHON || path.join(process.cwd(), ".venv-action", "bin", "python");
}

// All camera clip directories under a recording (streams/<node>/ holding camera_main.mp4).
function sessionClipDirs(sessionAbs: string): string[] {
  const streams = path.join(sessionAbs, "streams");
  let nodes: string[] = [];
  try {
    nodes = readdirSync(streams, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => path.join(streams, d.name));
  } catch {
    return [];
  }
  return nodes.filter((dir) => {
    try {
      return readdirSync(dir).includes("camera_main.mp4");
    } catch {
      return false;
    }
  });
}

function existingSlots(clipDirs: string[]): number[] {
  const found = new Set<number>();
  for (const dir of clipDirs) {
    for (const name of readdirSync(dir, { withFileTypes: true })) {
      const m = name.isDirectory() && /^analysis_(\d+)$/.exec(name.name);
      if (m) found.add(parseInt(m[1], 10));
    }
  }
  return [...found].sort((a, b) => a - b);
}

// POST — create (or re-run) an analysis slot for a recording.
// Body: { day, rec, slot?, settings:{stride,samplesPerClassify}, variants:["ntu"|"hmdb"],
//         disabled:{ action?: string[]; "action-hmdb"?: string[] } }
export async function POST(req: NextRequest) {
  let body: {
    day?: string;
    rec?: string;
    slot?: number;
    settings?: { stride?: number; samplesPerClassify?: number };
    variants?: string[];
    disabled?: Record<string, string[]>;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }
  const { day, rec } = body;
  if (!day || !rec) return NextResponse.json({ error: "day and rec required" }, { status: 400 });

  const sessionRel = `${day}/${rec}`;
  const sessionAbs = safeResolve(sessionRel);
  if (!sessionAbs) return NextResponse.json({ error: "bad path" }, { status: 400 });

  const clipDirs = sessionClipDirs(sessionAbs);
  if (clipDirs.length === 0) return NextResponse.json({ error: "no clips in recording" }, { status: 404 });

  const variants = (body.variants ?? ["ntu"]).filter((v) => v === "ntu" || v === "hmdb");
  if (variants.length === 0) return NextResponse.json({ error: "no valid variants" }, { status: 400 });

  // Allocate the slot. An explicit slot >=2 means "re-run this slot"; otherwise
  // pick max+1 and claim it atomically via mkdir (EEXIST => bump and retry), so
  // two concurrent creates can't grab the same number.
  let slot: number;
  if (body.slot && body.slot >= 2) {
    slot = Math.round(body.slot);
    for (const dir of clipDirs) mkdirSync(path.join(dir, `analysis_${slot}`), { recursive: true });
  } else {
    let candidate = Math.max(1, ...existingSlots(clipDirs)) + 1;
    for (let tries = 0; ; tries++, candidate++) {
      if (tries > 50) return NextResponse.json({ error: "could not allocate slot" }, { status: 500 });
      try {
        for (const dir of clipDirs) mkdirSync(path.join(dir, `analysis_${candidate}`)); // non-recursive: throws EEXIST
        break;
      } catch {
        // some clip already had this slot dir — clean up partial, bump, retry
        for (const dir of clipDirs) {
          try {
            rmSync(path.join(dir, `analysis_${candidate}`), { recursive: true, force: true });
          } catch {
            /* leave it; the bumped candidate will avoid it */
          }
        }
      }
    }
    slot = candidate;
  }

  // Snapshot the settings into each slot dir as a classes-config-shaped file, so
  // action.py reads stride/samplesPerClassify/whitelist straight from it.
  const config: Record<string, unknown> = {
    settings: {
      stride: Math.max(0, Math.round(body.settings?.stride ?? 0)),
      samplesPerClassify: Math.max(0, Math.round(body.settings?.samplesPerClassify ?? 0)),
    },
    variants,
    createdAt: new Date().toISOString(),
  };
  for (const v of ["action", "action-hmdb"]) {
    config[v] = { disabled: (body.disabled?.[v] ?? []).filter((s) => typeof s === "string") };
  }
  for (const dir of clipDirs) {
    writeFileSync(path.join(dir, `analysis_${slot}`, "config.json"), JSON.stringify(config, null, 2));
  }
  const configEnvPath = path.join(clipDirs[0], `analysis_${slot}`, "config.json");

  try {
    const { isolated, unit } = spawnBatch(
      actionPython(),
      [
        path.join(process.cwd(), "detect", "action.py"),
        "--session", sessionRel,
        "--slot", String(slot),
        "--variant", variants.join(","),
        "--force",
      ],
      { cwd: process.cwd(), logName: ".action.log", extraEnv: { SMARTROOM_ACTION_CLASSES_FILE: configEnvPath } },
    );
    return NextResponse.json({ started: true, slot, isolated, unit });
  } catch (e) {
    return NextResponse.json(
      { started: false, error: e instanceof Error ? e.message : "spawn failed" },
      { status: 500 },
    );
  }
}

// DELETE — remove a slot (>=2) from a recording. Query: ?path=<day>/<rec>&slot=N
export async function DELETE(req: NextRequest) {
  const rel = req.nextUrl.searchParams.get("path");
  const slot = Number(req.nextUrl.searchParams.get("slot"));
  if (!rel || !Number.isFinite(slot)) return NextResponse.json({ error: "path and slot required" }, { status: 400 });
  if (slot < 2) return NextResponse.json({ error: "cannot delete the original (slot 1)" }, { status: 400 });

  const sessionAbs = safeResolve(rel);
  if (!sessionAbs) return NextResponse.json({ error: "bad path" }, { status: 400 });

  const clipDirs = sessionClipDirs(sessionAbs);
  // Refuse if any clip's slot is mid-analysis (don't yank a running write).
  for (const dir of clipDirs) {
    const slotDir = path.join(dir, `analysis_${slot}`);
    let files: string[] = [];
    try {
      files = readdirSync(slotDir);
    } catch {
      continue;
    }
    for (const f of files) {
      if (!f.endsWith(".json") || !f.includes(".detections.")) continue;
      try {
        if (JSON.parse(readFileSync(path.join(slotDir, f), "utf8")).status === "analyzing") {
          return NextResponse.json({ error: "slot is still analyzing" }, { status: 409 });
        }
      } catch {
        /* ignore unreadable */
      }
    }
  }
  for (const dir of clipDirs) {
    rmSync(path.join(dir, `analysis_${slot}`), { recursive: true, force: true });
  }
  return NextResponse.json({ deleted: true, slot });
}
