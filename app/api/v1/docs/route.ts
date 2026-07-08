import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The API reference (API.md, kept in the repo next to the code it documents),
// served from the API itself so LAN consumers don't need repo access.
export async function GET() {
  try {
    const md = await readFile(path.join(process.cwd(), "API.md"), "utf8");
    return new NextResponse(md, {
      headers: {
        "content-type": "text/markdown; charset=utf-8",
        "access-control-allow-origin": "*",
        "cache-control": "no-store",
      },
    });
  } catch {
    return NextResponse.json({ error: "API.md not found" }, { status: 404 });
  }
}
