import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

import { spawnBatch } from "@/lib/spawn-batch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// validate.py is stdlib+ffprobe only, so the system python3 suffices.
// Override with SMARTROOM_VALIDATE_PYTHON if ever needed.
function validatePython() {
  return process.env.SMARTROOM_VALIDATE_PYTHON || "python3";
}

// Kick off data validation (detached, non-blocking). Same contract as
// /api/detect: the script holds a global flock so concurrent triggers are safe;
// { relPath } / { relPaths: [...] } validates a subset (neither = all clips).
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
  const script = path.join(projectRoot, "detect", "validate.py");
  const args = [script];
  for (const p of relPaths) args.push("--path", p);
  if (force || relPaths.length) args.push("--force");

  try {
    const { isolated, unit } = spawnBatch(validatePython(), args, {
      cwd: projectRoot,
      logName: ".validate.log",
    });
    return NextResponse.json({ started: true, isolated, unit });
  } catch (e) {
    return NextResponse.json(
      { started: false, error: e instanceof Error ? e.message : "spawn failed" },
      { status: 500 }
    );
  }
}
