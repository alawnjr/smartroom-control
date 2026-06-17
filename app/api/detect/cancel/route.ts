import { readFileSync, rmSync } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import { clearAnalyzing } from "@/lib/detections";
import { savedRoot } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Cancel an in-progress detection run: kill the process group recorded in
// recordings/.detect.pid (detect.py is a group leader, so this also kills its
// ffmpeg/forkserver children), then clear any stuck "analyzing" markers.
export async function POST() {
  const pidFile = path.join(savedRoot(), ".detect.pid");
  let cancelled = false;
  try {
    const pid = parseInt(readFileSync(pidFile, "utf8").trim(), 10);
    if (pid) {
      try {
        process.kill(-pid, "SIGKILL"); // whole group
        cancelled = true;
      } catch {
        try {
          process.kill(pid, "SIGKILL"); // fallback: just the leader
          cancelled = true;
        } catch {
          /* already gone */
        }
      }
    }
  } catch {
    /* no pid file / not running */
  }
  try {
    rmSync(pidFile, { force: true });
  } catch {
    /* ignore */
  }
  const cleared = clearAnalyzing();
  return NextResponse.json({ cancelled, cleared });
}
