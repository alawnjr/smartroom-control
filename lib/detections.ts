import { statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { savedRoot } from "@/lib/recordings";
import type { DetectionSummary } from "@/lib/types";

// Sidecar paths for a source mp4 (absolute): <stem>.detections.json and
// <stem>.annotated.mp4 — matches detect/detect.py.
export function sidecarPaths(absMp4: string) {
  const dir = path.dirname(absMp4);
  const stem = path.basename(absMp4, path.extname(absMp4));
  return {
    json: path.join(dir, `${stem}.detections.json`),
    annotated: path.join(dir, `${stem}.annotated.mp4`),
  };
}

// Read a clip's detection summary, applying the same staleness check as the
// Python script (stale/missing → "none"). Never throws.
export async function readDetectionSummary(absMp4: string): Promise<DetectionSummary> {
  const { json, annotated } = sidecarPaths(absMp4);
  try {
    const raw = JSON.parse(await readFile(json, "utf8"));
    const status = raw.status as DetectionSummary["status"];

    if (status === "error") {
      return { status: "error", error: raw.error ?? "analysis failed", hasAnnotated: false };
    }
    if (status === "analyzing") {
      return { status: "analyzing", hasAnnotated: false };
    }
    // status === "done": treat as stale (→ none) if the mp4 is newer
    const mp4Mtime = statSync(absMp4).mtimeMs;
    if ((raw.sourceMtimeMs ?? 0) < mp4Mtime) {
      return { status: "none", hasAnnotated: false };
    }
    let annotatedRelPath: string | undefined;
    let hasAnnotated = false;
    if (raw.hasAnnotated) {
      try {
        statSync(annotated);
        hasAnnotated = true;
        annotatedRelPath = path.relative(savedRoot(), annotated);
      } catch {
        hasAnnotated = false;
      }
    }
    return {
      status: "done",
      maxPersons: raw.maxPersons,
      avgPersons: raw.avgPersons,
      framesAnalyzed: raw.framesAnalyzed,
      timeline: raw.timeline,
      hasAnnotated,
      annotatedRelPath,
    };
  } catch {
    return { status: "none", hasAnnotated: false };
  }
}
