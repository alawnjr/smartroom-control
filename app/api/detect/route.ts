import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

import { spawnBatch } from "@/lib/spawn-batch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Default to the shared yolo-bench venv python; override with SMARTROOM_DETECT_PYTHON.
function detectPython() {
  return (
    process.env.SMARTROOM_DETECT_PYTHON ||
    path.join(process.env.HOME || "", "Code", "yolo-bench", ".venv", "bin", "python")
  );
}

// Kick off detection (detached, non-blocking). The Python script holds a global
// flock, so concurrent triggers (watcher / Save All / this button) are safe —
// extra runs exit immediately. Optional { relPath } re-analyzes a single clip;
// { relPaths: [...] } re-analyzes a selected subset in one run (neither = all).
export async function POST(req: NextRequest) {
  let relPaths: string[] = [];
  let force = false;
  try {
    const body = await req.json();
    const single = typeof body?.relPath === "string" ? [body.relPath] : [];
    const many = Array.isArray(body?.relPaths) ? body.relPaths.filter((x: unknown) => typeof x === "string") : [];
    relPaths = [...single, ...many];
    force = Boolean(body?.force);
  } catch {
    // no body
  }

  const projectRoot = process.cwd();
  const script = path.join(projectRoot, "detect", "detect.py");
  const args = [script];
  for (const p of relPaths) args.push("--path", p); // one run for the whole selection
  if (force || relPaths.length) args.push("--force"); // re-analyze forces a reprocess

  try {
    // Own cgroup via systemd-run so a control-panel restart can't kill an
    // in-flight batch — it runs start-to-finish. Logs to .detect.log.
    const { isolated, unit } = spawnBatch(detectPython(), args, {
      cwd: projectRoot,
      logName: ".detect.log",
    });
    return NextResponse.json({ started: true, isolated, unit });
  } catch (e) {
    return NextResponse.json(
      { started: false, error: e instanceof Error ? e.message : "spawn failed" },
      { status: 500 }
    );
  }
}
