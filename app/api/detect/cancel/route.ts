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
function killByPidFile(pidFile: string): boolean {
  let killed = false;
  try {
    const pid = parseInt(readFileSync(pidFile, "utf8").trim(), 10);
    if (pid) {
      try {
        process.kill(-pid, "SIGKILL"); // whole group
        killed = true;
      } catch {
        try {
          process.kill(pid, "SIGKILL"); // fallback: just the leader
          killed = true;
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
  return killed;
}

// Cancel both detection and action runs.
export async function POST() {
  const root = savedRoot();
  const a = killByPidFile(path.join(root, ".detect.pid"));
  const b = killByPidFile(path.join(root, ".action.pid"));
  const cleared = clearAnalyzing();
  return NextResponse.json({ cancelled: a || b, cleared });
}
