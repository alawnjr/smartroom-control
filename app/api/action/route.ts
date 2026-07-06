import { spawn } from "node:child_process";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function detectPython() {
  return (
    process.env.SMARTROOM_DETECT_PYTHON ||
    path.join(process.env.HOME || "", "Code", "yolo-bench", ".venv", "bin", "python")
  );
}

// Kick off per-person action recognition (detached, non-blocking). action.py
// holds its own flock, so concurrent triggers are safe. Optional { relPath }
// runs one clip; { force } reprocesses.
export async function POST(req: NextRequest) {
  let relPath: string | undefined;
  let force = false;
  try {
    const body = await req.json();
    relPath = body?.relPath;
    force = Boolean(body?.force);
  } catch {
    /* no body */
  }
  const projectRoot = process.cwd();
  const args = [path.join(projectRoot, "detect", "action.py")];
  if (relPath) args.push("--path", relPath);
  if (force || relPath) args.push("--force");

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
