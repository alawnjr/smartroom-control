import { spawn } from "node:child_process";
import { existsSync, openSync, readFileSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { NextResponse } from "next/server";

import { savedRoot } from "@/lib/recordings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Push the saved recordings to the public Vercel mirror (smartroom-mirror repo's
// scripts/sync.mjs — incremental: already-uploaded blobs are skipped). POST
// starts a sync; GET reports whether one is running + the last run's summary.

function mirrorDir() {
  return process.env.SMARTROOM_MIRROR_DIR || path.join(os.homedir(), "Code", "smartroom-mirror");
}

// The blob store token lives in the mirror repo's .env.local (vercel env pull);
// node doesn't auto-load that file, so read it here and hand it to the child.
function blobToken(): string | null {
  if (process.env.BLOB_READ_WRITE_TOKEN) return process.env.BLOB_READ_WRITE_TOKEN;
  try {
    const env = readFileSync(path.join(mirrorDir(), ".env.local"), "utf8");
    const m = env.match(/^\s*BLOB_READ_WRITE_TOKEN\s*=\s*"?([^"\n]+)"?\s*$/m);
    return m ? m[1].trim() : null;
  } catch {
    return null;
  }
}

const logPath = () => path.join(savedRoot(), ".mirror.log");
const pidPath = () => path.join(savedRoot(), ".mirror.pid");

function isRunning(): boolean {
  try {
    const pid = Number(readFileSync(pidPath(), "utf8").trim());
    if (!pid) return false;
    process.kill(pid, 0); // throws when the process is gone
    return true;
  } catch {
    return false;
  }
}

// The final line sync.mjs prints, e.g.
// "manifest: 12 sessions | uploaded 34 blobs (56.7 MB) | 890 up-to-date"
function lastRun() {
  try {
    const log = readFileSync(logPath(), "utf8");
    const m = log.match(/^manifest: .*$/m);
    const lines = log.trimEnd().split("\n");
    return {
      summary: m ? m[0] : null,
      tail: lines.slice(-3),
      finishedAt: statSync(logPath()).mtimeMs,
      failed: !m && !isRunning() && log.length > 0,
    };
  } catch {
    return { summary: null, tail: [], finishedAt: null, failed: false };
  }
}

export async function GET() {
  return NextResponse.json({ running: isRunning(), ...lastRun() });
}

export async function POST() {
  if (isRunning()) {
    return NextResponse.json({ started: false, running: true });
  }
  const script = path.join(mirrorDir(), "scripts", "sync.mjs");
  if (!existsSync(script)) {
    return NextResponse.json(
      { started: false, error: `mirror repo not found (${script})` },
      { status: 500 },
    );
  }
  const token = blobToken();
  if (!token) {
    return NextResponse.json(
      { started: false, error: "no BLOB_READ_WRITE_TOKEN (put it in the mirror repo's .env.local)" },
      { status: 500 },
    );
  }
  const fd = openSync(logPath(), "w"); // fresh log per run — GET parses it
  const child = spawn(process.execPath, [script, "--root", savedRoot()], {
    cwd: mirrorDir(),
    detached: true,
    stdio: ["ignore", fd, fd],
    env: { ...process.env, BLOB_READ_WRITE_TOKEN: token },
  });
  writeFileSync(pidPath(), String(child.pid ?? ""));
  child.unref();
  return NextResponse.json({ started: true });
}
