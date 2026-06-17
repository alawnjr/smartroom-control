import { readdirSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { savedRoot } from "@/lib/recordings";
import type { DetectionSummary } from "@/lib/types";

// Per-model sidecars for a source mp4 (matches detect/detect.py):
//   <stem>.detections.<model>.json  and  <stem>.annotated.<model>.mp4
function detectionsPrefix(absMp4: string) {
  const stem = path.basename(absMp4, path.extname(absMp4));
  return `${stem}.detections.`;
}

function annotatedName(absMp4: string, model: string) {
  const stem = path.basename(absMp4, path.extname(absMp4));
  return `${stem}.annotated.${model}.mp4`;
}

async function readOne(jsonPath: string, absMp4: string, model: string): Promise<DetectionSummary> {
  try {
    const raw = JSON.parse(await readFile(jsonPath, "utf8"));
    const status = raw.status as DetectionSummary["status"];
    if (status === "error") return { model, status: "error", error: raw.error ?? "analysis failed", hasAnnotated: false };
    if (status === "analyzing") return { model, status: "analyzing", hasAnnotated: false };

    // done: stale (→ none) if the source mp4 is newer
    if ((raw.sourceMtimeMs ?? 0) < statSync(absMp4).mtimeMs) {
      return { model, status: "none", hasAnnotated: false };
    }
    let hasAnnotated = false;
    let annotatedRelPath: string | undefined;
    if (raw.hasAnnotated) {
      const annotated = path.join(path.dirname(absMp4), annotatedName(absMp4, model));
      try {
        statSync(annotated);
        hasAnnotated = true;
        annotatedRelPath = path.relative(savedRoot(), annotated);
      } catch {
        hasAnnotated = false;
      }
    }
    return {
      model,
      status: "done",
      maxPersons: raw.maxPersons,
      avgPersons: raw.avgPersons,
      framesAnalyzed: raw.framesAnalyzed,
      timeline: raw.timeline,
      hasAnnotated,
      annotatedRelPath,
    };
  } catch {
    return { model, status: "none", hasAnnotated: false };
  }
}

// All models' summaries for a clip, keyed by model. Empty object if none.
export async function readDetections(absMp4: string): Promise<Record<string, DetectionSummary>> {
  const dir = path.dirname(absMp4);
  const prefix = detectionsPrefix(absMp4);
  let names: string[] = [];
  try {
    names = readdirSync(dir);
  } catch {
    return {};
  }
  const out: Record<string, DetectionSummary> = {};
  for (const name of names) {
    if (!name.startsWith(prefix) || !name.endsWith(".json")) continue;
    const model = name.slice(prefix.length, -".json".length);
    if (!model) continue;
    out[model] = await readOne(path.join(dir, name), absMp4, model);
  }
  return out;
}
