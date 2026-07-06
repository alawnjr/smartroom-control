import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Shared with detect/action.py (SMARTROOM_ACTION_CLASSES_FILE). A JSON map of
// variant key -> { disabled: [class name, ...] }. action.py masks disabled
// classes at inference; this route is the dashboard's read/write of that file.
function configPath() {
  return (
    process.env.SMARTROOM_ACTION_CLASSES_FILE ||
    path.join(process.cwd(), "action-classes.json")
  );
}

type Config = Record<string, { disabled: string[] }>;

async function read(): Promise<Config> {
  try {
    return JSON.parse(await readFile(configPath(), "utf8"));
  } catch {
    return {};
  }
}

export async function GET() {
  return NextResponse.json(await read());
}

// Body: { variant: string, disabled: string[] } updates one variant, or a full
// Config object to replace everything.
export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }

  const cfg = await read();
  if (body && typeof body === "object" && "variant" in body) {
    const { variant, disabled } = body as { variant: string; disabled: string[] };
    if (typeof variant !== "string" || !Array.isArray(disabled)) {
      return NextResponse.json({ error: "expected { variant, disabled[] }" }, { status: 400 });
    }
    cfg[variant] = { disabled: disabled.filter((d) => typeof d === "string") };
  } else if (body && typeof body === "object") {
    Object.assign(cfg, body as Config);
  } else {
    return NextResponse.json({ error: "expected object" }, { status: 400 });
  }

  await writeFile(configPath(), JSON.stringify(cfg, null, 2));
  return NextResponse.json({ ok: true, config: cfg });
}
