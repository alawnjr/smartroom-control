import { spawn } from "node:child_process";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

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
// extra runs exit immediately. Optional { relPath } re-analyzes a single clip.
export async function POST(req: NextRequest) {
  let relPath: string | undefined;
  try {
    relPath = (await req.json())?.relPath;
  } catch {
    // no body
  }

  const projectRoot = process.cwd();
  const script = path.join(projectRoot, "detect", "detect.py");
  const args = [script];
  if (relPath) args.push("--path", relPath, "--force");

  try {
    const child = spawn(detectPython(), args, {
      cwd: projectRoot,
      detached: true,
      stdio: "ignore",
      env: process.env,
    });
    child.unref();
    return NextResponse.json({ started: true });
  } catch (e) {
    return NextResponse.json(
      { started: false, error: e instanceof Error ? e.message : "spawn failed" },
      { status: 500 }
    );
  }
}
