import { spawn } from "node:child_process";
import { existsSync, openSync } from "node:fs";
import path from "node:path";

import { savedRoot } from "@/lib/recordings";

// The detect/action batches are launched from this Next.js process, which itself
// runs as a systemd *user* service. A plain `detached + unref()` child still
// lives in the service's cgroup, so restarting smartroom-control.service (every
// rebuild, or Restart=on-failure) kills the in-flight batch mid-run — the classic
// "it only analyzed 1 or 2 clips then stopped". To make a batch run start-to-
// finish uninterrupted, we launch it in its OWN transient cgroup via
// `systemd-run --user`, which is fully decoupled from this server's lifecycle.
//
// Falls back to a detached spawn when systemd-run isn't available (non-systemd
// host, or no user bus) — same behaviour as before, just not crash-isolated.
function systemdRunAvailable(): boolean {
  return (
    existsSync("/usr/bin/systemd-run") &&
    Boolean(process.env.XDG_RUNTIME_DIR)
  );
}

// Env the batch actually needs: the SMARTROOM_* overrides plus HOME/PATH (for
// ~/Code default checkpoint paths and ffmpeg on PATH). systemd-run --user starts
// the transient unit with the service manager's env, not ours, so forward these
// explicitly with --setenv.
function forwardedEnv(): string[] {
  const out: string[] = [];
  for (const [k, v] of Object.entries(process.env)) {
    if (v === undefined) continue;
    if (k === "HOME" || k === "PATH" || k.startsWith("SMARTROOM_")) {
      out.push("--setenv", `${k}=${v}`);
    }
  }
  return out;
}

// Launch a long-running batch (detect.py / action.py) so it survives restarts of
// this web server. `logName` is a file under the recordings root that captures
// the batch's stdout+stderr for diagnosis (e.g. ".action.log").
export function spawnBatch(
  python: string,
  args: string[],
  opts: { logName: string; cwd: string; extraEnv?: Record<string, string> }
) {
  const logPath = path.join(savedRoot(), opts.logName);
  const extraEnv = opts.extraEnv ?? {};

  if (systemdRunAvailable()) {
    const unit = `smartroom-${opts.logName.replace(/[^a-z0-9]+/gi, "")}-${Date.now()}`;
    const sdArgs = [
      "--user",
      "--collect", // remove the transient unit once it exits
      "--quiet",
      `--unit=${unit}`,
      "--description=smartroom analysis batch",
      `--working-directory=${opts.cwd}`,
      ...forwardedEnv(),
      ...Object.entries(extraEnv).flatMap(([k, v]) => ["--setenv", `${k}=${v}`]),
      `--property=StandardOutput=append:${logPath}`,
      `--property=StandardError=append:${logPath}`,
      "--",
      python,
      ...args,
    ];
    const child = spawn("/usr/bin/systemd-run", sdArgs, {
      cwd: opts.cwd,
      stdio: "ignore",
      env: process.env,
    });
    child.unref();
    return { isolated: true as const, unit };
  }

  // Fallback: detached spawn (not cgroup-isolated). Still redirect output to the
  // log file rather than discarding it, so failures are diagnosable.
  const fd = openSync(logPath, "a");
  const child = spawn(python, args, {
    cwd: opts.cwd,
    detached: true,
    stdio: ["ignore", fd, fd],
    env: { ...process.env, ...extraEnv },
  });
  child.unref();
  return { isolated: false as const, unit: null };
}
