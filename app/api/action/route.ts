import { spawn } from "node:child_process";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The action pipeline needs the mmcv/mmaction2 stack, which lives in the
// dedicated Python 3.10 venv (.venv-action), not the py3.14 detection venv.
function actionPython() {
  return (
    process.env.SMARTROOM_ACTION_PYTHON ||
    path.join(process.cwd(), ".venv-action", "bin", "python")
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
    const child = spawn(actionPython(), args, {
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
