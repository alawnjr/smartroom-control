// Room-map card: top-down plan view of a RECORDING — all located cameras'
// tracks merged onto the shared AprilTag room frame (every camera's extrinsics
// are relative to tag 1, so their positions live in one coordinate system).
// Positions come from the per-clip centroids sidecars: "geo" (localize.py,
// pose + depth) preferred, the action pass's sidecars as fallback for old
// recordings.
//
// Rendered as one card in the session grid; returns null when no clip has any
// centroids sidecar, and removes itself quietly if nothing turns out located.

import { h, fileUrl } from "./api.js";

// One hue family per camera so "which camera saw this" reads at a glance.
const CAM_PALETTES = [
  ["#10b981", "#34d399", "#059669", "#6ee7b7"], // greens
  ["#f59e0b", "#fb923c", "#d97706", "#fcd34d"], // oranges
  ["#3b82f6", "#60a5fa", "#2563eb", "#93c5fd"], // blues
  ["#a855f7", "#c084fc", "#9333ea", "#d8b4fe"], // purples
];
const CAM_MARKERS = ["#0ea5e9", "#e879f9", "#22d3ee", "#f472b6"];

const cache = new Map(); // relPath@version -> centroids JSON (or null)

// The centroids source for one clip: prefer the dedicated localization pass,
// fall back to the action models so pre-geo recordings keep their maps.
function centroidsInfo(v) {
  for (const m of ["geo", "action", "action-hmdb"]) {
    const d = v.detections?.[m];
    if (d?.status === "done") return { path: v.relPath.replace(/\.mp4$/, `.centroids.${m}.json`), version: d.version };
  }
  return null;
}

function fetchCentroids(info) {
  const key = `${info.path}@${info.version ?? 0}`;
  if (cache.has(key)) return Promise.resolve(cache.get(key));
  return fetch(fileUrl(info.path, info.version))
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null)
    .then((data) => { cache.set(key, data); return data; });
}

// sources: [{ data (centroids json with roomFrame), label }]
function buildSVG(sources) {
  const cams = sources.map((s, ci) => ({
    ...s,
    ci,
    cam: s.data.roomFrame.cameraPositionMm ?? [0, 0, 0],
    tracks: Object.entries(s.data.persons ?? {})
      .map(([tid, entries]) => ({ tid, pts: entries.filter((e) => Array.isArray(e.room)) }))
      .filter((tr) => tr.pts.length >= 2),
  })).filter((c) => c.tracks.length);
  if (!cams.length) return null;

  const xs = [0], zs = [0];
  for (const c of cams) {
    xs.push(c.cam[0]);
    zs.push(c.cam[2]);
    for (const tr of c.tracks) for (const p of tr.pts) { xs.push(p.room[0]); zs.push(p.room[1]); }
  }
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
  // the tag's wall (Z = 0) + tag marker
  s += `<line x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(0)}" stroke="var(--muted)" stroke-width="1.5"/>`;
  s += `<rect x="${sx(0) - 5}" y="${sy(0) - 3}" width="10" height="6" rx="1" fill="#e11d48"/>`;
  const legend = [];
  for (const c of cams) {
    const mk = CAM_MARKERS[c.ci % CAM_MARKERS.length];
    // camera marker with a short sight-line toward the tag
    s += `<g transform="translate(${sx(c.cam[0])},${sy(c.cam[2])})"><circle r="5" fill="${mk}"/>` +
      `<line x1="0" y1="0" x2="${(sx(0) - sx(c.cam[0])) * 0.25}" y2="${(sy(0) - sy(c.cam[2])) * 0.25}" stroke="${mk}" stroke-width="1.5"/></g>`;
    const palette = CAM_PALETTES[c.ci % CAM_PALETTES.length];
    legend.push({ color: mk, text: `${c.label} camera` });
    c.tracks.forEach((tr, i) => {
      const color = palette[i % palette.length];
      const d = tr.pts.map((p, j) => `${j ? "L" : "M"}${sx(p.room[0]).toFixed(1)},${sy(p.room[1]).toFixed(1)}`).join("");
      const first = tr.pts[0], last = tr.pts[tr.pts.length - 1];
      s += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6" stroke-opacity=".75" stroke-linejoin="round" stroke-linecap="round"/>`;
      s += `<circle cx="${sx(first.room[0])}" cy="${sy(first.room[1])}" r="2.5" fill="${color}" fill-opacity=".5"/>`;
      s += `<circle cx="${sx(last.room[0])}" cy="${sy(last.room[1])}" r="3.5" fill="${color}"/>`;
      const depthPts = tr.pts.filter((p) => (p.src || "").startsWith("depth")).length;
      legend.push({ color, text: `${c.label} · person ${tr.tid} (${tr.pts.length} pts${depthPts ? `, ${depthPts} depth` : ""})` });
    });
  }

  const svg = document.createElement("div");
  svg.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${s}</svg>`;
  return { svg: svg.firstChild, legend, tagId: cams[0].data.roomFrame.tagId };
}

// "D455" / "D435" from the filename, else the node name (webcam clips).
function camLabel(v, nameFor) {
  const m = v.file.match(/camera_(d\d+)_color/);
  return m ? m[1].toUpperCase() : nameFor(v.node);
}

// ONE merged map card for a whole recording. Returns a placeholder element
// that fills itself in (or removes itself) once the sidecars arrive; null when
// no clip in the session has a centroids-bearing analysis.
export function sessionMapCard(clips, nameFor) {
  const sources = clips
    .map((v) => ({ v, info: centroidsInfo(v) }))
    .filter((s) => s.info);
  if (!sources.length) return null;
  const card = h("div", { class: "card card-sm map-card" });

  Promise.all(sources.map((s) => fetchCentroids(s.info))).then((datas) => {
    const located = datas
      .map((data, i) => ({ data, label: camLabel(sources[i].v, nameFor) }))
      .filter((s) => s.data?.roomFrame);
    const built = located.length ? buildSVG(located) : null;
    if (!built) return void card.remove();
    card.replaceChildren(
      h("div", { class: "hd" },
        h("b", {}, "room map"),
        h("span", {}, `tag ${built.tagId ?? "?"} at origin · grid 1 m`)),
      built.svg,
      h("div", { class: "map-legend" },
        h("span", {}, h("i", { class: "sw sw-tag" }), " tag"),
        ...built.legend.map((l) =>
          h("span", {}, h("i", { class: "sw", style: `background:${l.color}` }), ` ${l.text}`))));
  });
  return card;
}
