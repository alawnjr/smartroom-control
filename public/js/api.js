// Shared fetch + DOM helpers for the vanilla frontend.

export async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

export function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).catch(() => {});
}

export function fileUrl(relPath, version) {
  const base = `/api/saved/file?path=${encodeURIComponent(relPath)}`;
  return version ? `${base}&v=${version}` : base;
}

// h("button", {class: "tbtn", onclick: fn, title: "..."}, "label", childEl, ...)
// null/undefined/false children are skipped; strings become text nodes.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k in el && k !== "list" && typeof v === "boolean") el[k] = v;
    else el.setAttribute(k, v);
  }
  el.append(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}

export function svgEl(markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  return t.content.firstChild;
}

export function fmtClock(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function fmtDur(s) {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${String(Math.round(s % 60)).padStart(2, "0")}s`;
}

// A recording "session" = clips captured together (same day/rec across cameras).
// Newest first. Ported from lib/use-saved.ts.
export function groupSessions(videos) {
  const map = new Map();
  for (const v of videos) {
    const key = `${v.day}/${v.rec}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(v);
  }
  const sessions = [...map.entries()].map(([key, clips]) => {
    const { day, rec } = clips[0];
    const date = day.replace(/^day_\d+_/, "");
    const take = rec.split("_").pop() ?? rec;
    return {
      key,
      label: `${date} · take ${take}`,
      clips: [...clips].sort((a, b) => a.node.localeCompare(b.node)),
      mtime: Math.max(...clips.map((c) => c.mtime)),
    };
  });
  sessions.sort((a, b) => b.mtime - a.mtime);
  return sessions;
}

export function clipAnalyzing(v) {
  return Object.values(v.detections ?? {}).some((d) => d.status === "analyzing");
}

export function analyzingCount(videos) {
  return videos.filter(clipAnalyzing).length;
}

export function validatingCount(videos) {
  return videos.filter((v) => v.validation?.status === "analyzing").length;
}
