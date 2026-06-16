import path from "node:path";

// Root folder where "Save All" writes (and where the gallery reads) — a
// gitignored recordings/ inside the project, overridable with SMARTROOM_SAVE_DIR.
// Kept in sync with app/api/save-all/route.ts.
export function savedRoot(): string {
  return process.env.SMARTROOM_SAVE_DIR || path.join(process.cwd(), "recordings");
}

// Resolve a client-supplied relative path under savedRoot(), refusing anything
// that escapes the root (path-traversal guard). Returns null if unsafe.
export function safeResolve(rel: string): string | null {
  const root = path.resolve(savedRoot());
  const abs = path.resolve(root, rel);
  if (abs !== root && !abs.startsWith(root + path.sep)) return null;
  return abs;
}
