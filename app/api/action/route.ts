import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

import { spawnBatch } from "@/lib/spawn-batch";

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
  let relPaths: string[] = [];
  let force = false;
  let variant = "ntu";
  try {
    const body = await req.json();
    const single = typeof body?.relPath === "string" ? [body.relPath] : [];
    const many = Array.isArray(body?.relPaths) ? body.relPaths.filter((x: unknown) => typeof x === "string") : [];
    relPaths = [...single, ...many];
    force = Boolean(body?.force);
    if (body?.variant === "hmdb") variant = "hmdb";
  } catch {
    /* no body */
  }
  const projectRoot = process.cwd();
  const args = [path.join(projectRoot, "detect", "action.py"), "--variant", variant];
  for (const p of relPaths) args.push("--path", p); // one run for the whole selection
  if (force || relPaths.length) args.push("--force");

  try {
    // Launched in its own cgroup (systemd-run) so a control-panel restart does
    // not kill an in-flight batch — runs start-to-finish. Logs to .action.log.
    const { isolated, unit } = spawnBatch(actionPython(), args, {
      cwd: projectRoot,
      logName: ".action.log",
    });
    return NextResponse.json({ started: true, isolated, unit });
  } catch (e) {
    return NextResponse.json(
      { started: false, error: e instanceof Error ? e.message : "spawn failed" },
      { status: 500 }
    );
  }
}
