// Room-map side-card: top-down plan view of each located clip (port of the old
// room-map.tsx ClipMap). Positions come from the centroids sidecar written by
// detect/action.py — entries carry room:[x,z] mm relative to the AprilTag's
// floor point when the clip has intrinsic + extrinsic calibration.
//
// Rendered as a card NEXT TO the clip's video card in the session grid; returns
// null immediately when no action sidecar exists, and removes itself quietly if
// the fetched sidecar has no roomFrame (clip not located).

import { h, fileUrl } from "./api.js";

const TRACK_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];
const cache = new Map(); // relPath@version -> centroids JSON (or null)

function centroidsInfo(v) {
  for (const m of ["action", "action-hmdb"]) {
    const d = v.detections?.[m];
    if (d?.status === "done") return { path: v.relPath.replace(/\.mp4$/, `.centroids.${m}.json`), version: d.version };
  }
  return null;
}

function buildSVG(data) {
  const cam = data.roomFrame.cameraPositionMm ?? [0, 0, 0];
  const tracks = Object.entries(data.persons ?? {})
    .map(([tid, entries]) => ({ tid, pts: entries.filter((e) => Array.isArray(e.room)) }))
    .filter((tr) => tr.pts.length >= 2);
  if (!tracks.length) return null;

  const xs = [0, cam[0]], zs = [0, cam[2]];
  for (const tr of tracks) for (const p of tr.pts) { xs.push(p.room[0]); zs.push(p.room[1]); }
  const pad = 400;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minZ = Math.min(...zs) - pad, maxZ = Math.max(...zs) + pad;
  const W = 340, H = Math.max(220, Math.min(420, (W * (maxZ - minZ)) / (maxX - minX)));
  const sx = (x) => ((x - minX) / (maxX - minX)) * W;
  const sy = (z) => ((z - minZ) / (maxZ - minZ)) * H;

  let s = "";
  for (let g = Math.ceil(minX / 1000) * 1000; g <= maxX; g += 1000)
    s += `<line x1="${sx(g)}" y1="0" x2="${sx(g)}" y2="${H}" stroke="var(--line)" stroke-width=".5"/>`;
  for (let g = Math.ceil(minZ / 1000) * 1000; g <= maxZ; g += 1000)
    s += `<line x1="0" y1="${sy(g)}" x2="${W}" y2="${sy(g)}" stroke="var(--line)" stroke-width=".5"/>`;
  // the tag's wall (Z = 0) + tag marker + camera marker
  s += `<line x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(0)}" stroke="var(--muted)" stroke-width="1.5"/>`;
  s += `<rect x="${sx(0) - 5}" y="${sy(0) - 3}" width="10" height="6" rx="1" fill="#e11d48"/>`;
  s += `<g transform="translate(${sx(cam[0])},${sy(cam[2])})"><circle r="5" fill="#0ea5e9"/><line x1="0" y1="0" x2="${(sx(0) - sx(cam[0])) * 0.25}" y2="${(sy(0) - sy(cam[2])) * 0.25}" stroke="#0ea5e9" stroke-width="1.5"/></g>`;
  tracks.forEach((tr, i) => {
    const color = TRACK_COLORS[i % TRACK_COLORS.length];
    const d = tr.pts.map((p, j) => `${j ? "L" : "M"}${sx(p.room[0]).toFixed(1)},${sy(p.room[1]).toFixed(1)}`).join("");
    const first = tr.pts[0], last = tr.pts[tr.pts.length - 1];
    s += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6" stroke-opacity=".75" stroke-linejoin="round" stroke-linecap="round"/>`;
    s += `<circle cx="${sx(first.room[0])}" cy="${sy(first.room[1])}" r="2.5" fill="${color}" fill-opacity=".5"/>`;
    s += `<circle cx="${sx(last.room[0])}" cy="${sy(last.room[1])}" r="3.5" fill="${color}"/>`;
  });

  const svg = document.createElement("div");
  svg.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${s}</svg>`;
  return { svg: svg.firstChild, tracks };
}

// Returns a placeholder card element that fills itself in (or removes itself)
// once the centroids sidecar arrives. Null when the clip has no action analysis.
export function roomMapCard(v, roomName) {
  const info = centroidsInfo(v);
  if (!info) return null;
  const key = `${info.path}@${info.version ?? 0}`;
  const card = h("div", { class: "card card-sm map-card" });

  const fill = (data) => {
    if (!data?.roomFrame) return void card.remove();
    const built = buildSVG(data);
    if (!built) return void card.remove();
    card.replaceChildren(
      h("div", { class: "hd" },
        h("b", {}, `${roomName} · room map`),
        h("span", {}, `tag ${data.roomFrame.tagId ?? "?"} at origin · grid 1 m`)),
      built.svg,
      h("div", { class: "map-legend" },
        h("span", {}, h("i", { class: "sw sw-tag" }), " tag"),
        h("span", {}, h("i", { class: "sw", style: "background:#0ea5e9" }), " camera"),
        ...built.tracks.map((tr, i) =>
          h("span", {}, h("i", { class: "sw", style: `background:${TRACK_COLORS[i % TRACK_COLORS.length]}` }), ` person ${tr.tid} (${tr.pts.length} pts)`))));
  };

  if (cache.has(key)) {
    const data = cache.get(key);
    if (!data?.roomFrame) return null; // known unlocated — don't even flash a card
    fill(data);
  } else {
    fetch(fileUrl(info.path, info.version))
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { cache.set(key, data); fill(data); })
      .catch(() => card.remove());
  }
  return card;
}
