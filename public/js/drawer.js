// Right slide-over for one clip's per-person action analysis: video (speed +
// raw/labels toggle) driving live top-class bars and a probability-over-time
// line graph with a playhead. Port of clip-analytics-drawer.tsx + action-bars.tsx.
//
// Static geometry (lines, gridlines, bar rows) is built once; each animation
// frame only moves the playhead and bar widths.

import { h, fileUrl } from "./api.js";

const SPEEDS = [0.25, 0.5, 1, 2];
const TOP_N = 5;
const SCALE = 0.4; // probability mapped to a full-width bar
const LW = 320, LH = 150, PAD = { l: 30, r: 8, t: 8, b: 26 };

const entryTop = (e) => e.top ?? [[e.action, e.conf]];

// Blend the two windows straddling t so bars glide instead of snapping.
function distAt(entries, t) {
  let i = -1;
  for (let k = 0; k < entries.length; k++) {
    if (entries[k].t <= t) i = k;
    else break;
  }
  if (i < 0) return null;
  const prev = entries[i], next = entries[i + 1];
  const frac = next && next.t > prev.t ? Math.min(1, Math.max(0, (t - prev.t) / (next.t - prev.t))) : 0;
  const dist = new Map();
  const add = (pairs, w) => { for (const [l, p] of pairs) dist.set(l, (dist.get(l) ?? 0) + p * w); };
  add(entryTop(prev), 1 - frac);
  if (next) add(entryTop(next), frac);
  return { dist, ref: frac >= 0.5 && next ? next : prev };
}

function barClass(label, actions) {
  const i = actions.indexOf(label);
  return i >= 0 ? `bar-${i % 5}` : "bar-x";
}

// One person's panel; returns {el, update(t)}.
function personPanel(id, entries, actions) {
  const peak = new Map();
  for (const e of entries) for (const [l, p] of entryTop(e)) peak.set(l, Math.max(peak.get(l) ?? 0, p));
  const rows = [...peak.entries()].sort((a, b) => b[1] - a[1]).slice(0, TOP_N).map(([l]) => l);

  const headerLab = h("span", { class: "lab" }, "idle");
  const bars = rows.map((label) => {
    const fill = h("div", { class: `bfill ${barClass(label, actions)}`, style: "width:0%" });
    const val = h("span", { class: "bval" }, "0.00");
    const row = h("div", { class: "brow" },
      h("span", { class: "bl", title: label }, label),
      h("div", { class: "btrack" }, fill),
      val);
    return { label, fill, val, row };
  });

  // ----- line graph (built once) -----
  let lineSVG = null, playhead = null, xScale = null, tMax = 1;
  const lineHost = h("div", {});
  if (entries.length >= 2) {
    tMax = entries[entries.length - 1].t || 1;
    const probAt = (e, label) => entryTop(e).find(([l]) => l === label)?.[1] ?? 0;
    const labels = [...peak.entries()].sort((a, b) => b[1] - a[1]).map(([l]) => l);
    const colorAt = (i) => `hsl(${Math.round((i * 137.508) % 360)} 65% 45%)`;
    let pMax = 0.3;
    for (const l of labels) for (const e of entries) pMax = Math.max(pMax, probAt(e, l));
    pMax = Math.ceil(pMax * 10) / 10;
    const x0 = PAD.l, x1 = LW - PAD.r, y0 = LH - PAD.b, y1 = PAD.t;
    const x = (t) => x0 + (t / tMax) * (x1 - x0);
    const y = (p) => y0 + (1 - p / pMax) * (y1 - y0);
    xScale = x;

    let s = "";
    for (const p of [0, pMax / 2, pMax])
      s += `<line x1="${x0}" x2="${x1}" y1="${y(p)}" y2="${y(p)}" stroke="rgb(0 0 0 / .08)"/><text x="${x0 - 4}" y="${y(p) + 3}" font-size="9" text-anchor="end" fill="rgb(0 0 0 / .4)">${p.toFixed(1)}</text>`;
    for (const t of [0, tMax / 2, tMax])
      s += `<text x="${x(t)}" y="${y0 + 13}" font-size="9" text-anchor="middle" fill="rgb(0 0 0 / .4)">${t.toFixed(t < 10 ? 1 : 0)}</text>`;
    s += `<line x1="${x0}" x2="${x1}" y1="${y0}" y2="${y0}" stroke="rgb(0 0 0 / .25)"/><line x1="${x0}" x2="${x0}" y1="${y1}" y2="${y0}" stroke="rgb(0 0 0 / .25)"/>`;
    s += `<text x="${(x0 + x1) / 2}" y="${LH - 2}" font-size="9" text-anchor="middle" fill="rgb(0 0 0 / .5)">time (s)</text>`;
    s += `<text x="9" y="${(y0 + y1) / 2}" font-size="9" text-anchor="middle" fill="rgb(0 0 0 / .5)" transform="rotate(-90 9 ${(y0 + y1) / 2})">probability</text>`;
    const legend = [];
    labels.forEach((l, i) => {
      const pts = entries.map((e) => `${x(e.t).toFixed(1)},${y(probAt(e, l)).toFixed(1)}`).join(" ");
      s += `<polyline points="${pts}" fill="none" stroke="${colorAt(i)}" stroke-width="1.75" stroke-linejoin="round"/>`;
      legend.push(h("span", {}, h("i", { style: `background:${colorAt(i)}` }), l));
    });
    s += `<line class="ph" x1="${x0}" x2="${x0}" y1="${y1}" y2="${y0}" stroke="rgb(0 0 0 / .5)" stroke-dasharray="3 2"/>`;

    const div = document.createElement("div");
    div.innerHTML = `<svg viewBox="0 0 ${LW} ${LH}" style="width:100%">${s}</svg>`;
    lineSVG = div.firstChild;
    playhead = lineSVG.querySelector(".ph");
    lineHost.append(h("div", { class: "legend-lines" }, ...legend), lineSVG);
  }

  const el = h("div", { class: "person-panel" },
    h("div", { class: "php" }, h("span", {}, `#${id}`), headerLab),
    h("div", {}, ...bars.map((b) => b.row)),
    lineHost);

  const update = (t) => {
    const at = distAt(entries, t);
    const kept = !!at && at.ref.kept !== false;
    headerLab.textContent = at ? at.ref.action : "idle";
    headerLab.className = `lab ${kept ? "on" : ""}`;
    for (const b of bars) {
      const p = at?.dist.get(b.label) ?? 0;
      b.fill.style.width = `${Math.min(100, (p / SCALE) * 100)}%`;
      b.val.textContent = p.toFixed(2);
    }
    if (playhead && xScale) {
      const px = xScale(Math.min(t, tMax));
      playhead.setAttribute("x1", px);
      playhead.setAttribute("x2", px);
    }
  };
  return { el, update };
}

export function openDrawer(v, model, d, roomName) {
  const host = document.getElementById("drawer-host");
  const hasOverlay = Boolean(d.hasAnnotated && d.annotatedRelPath);
  let overlay = hasOverlay;
  let rate = 1;
  let raf = 0;
  const panels = [];

  const close = () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("keydown", onKey);
    host.replaceChildren();
  };
  const onKey = (e) => e.key === "Escape" && close();
  window.addEventListener("keydown", onKey);

  const video = h("video", { controls: true, autoplay: true, preload: "auto" });
  const src = () => (overlay && hasOverlay ? fileUrl(d.annotatedRelPath, d.version) : fileUrl(v.relPath));
  video.src = src();

  const update = () => panels.forEach((p) => p.update(video.currentTime));
  const tick = () => { update(); raf = requestAnimationFrame(tick); };
  video.addEventListener("playing", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); });
  for (const ev of ["pause", "ended", "seeked", "timeupdate"]) video.addEventListener(ev, () => { if (video.paused) { cancelAnimationFrame(raf); update(); } });

  const obtn = hasOverlay ? h("button", { class: "vbtn right", onclick: () => {
    overlay = !overlay;
    const t = video.currentTime;
    video.src = src();
    video.currentTime = t;
    video.playbackRate = rate;
    obtn.textContent = overlay ? "raw" : "labels";
    video.play().catch(() => {});
  } }, "raw") : null;

  const speedSeg = h("div", { class: "seg" }, ...SPEEDS.map((r) => {
    const b = h("button", { class: r === 1 ? "on" : "", onclick: () => {
      rate = r;
      video.playbackRate = r;
      [...speedSeg.children].forEach((c) => c.classList.toggle("on", c === b));
    } }, `${r}×`);
    return b;
  }));

  const tags = (d.actions ?? []).slice(0, 8);
  const panelHost = h("div", {});

  host.replaceChildren(
    h("div", { class: "drawer-veil", onclick: close }),
    h("div", { class: "drawer" },
      h("div", { class: "hd" },
        h("div", {}, roomName, " ", h("span", { class: "take" }, `· ${v.rec.split("_").pop()}`)),
        h("button", { class: "icobtn", title: "Close", onclick: close }, "✕")),
      h("div", { class: "vwrap" }, video, obtn),
      h("div", { class: "speed-row" }, "speed", speedSeg),
      tags.length ? h("div", { class: "tags", style: "margin-top:8px" }, ...tags.map((t, i) => h("span", { class: `tag tag-${i % 5}` }, t))) : null,
      panelHost));

  // Load the per-window timeline and build one panel per tracked person.
  const path = d.actionsRelPath ?? v.relPath.replace(/\.mp4$/, `.actions.${model}.json`);
  fetch(fileUrl(path, d.version)).then((r) => (r.ok ? r.json() : null)).then((data) => {
    const tracks = data?.tracks ?? {};
    const ids = Object.keys(tracks).sort((a, b) => Number(a) - Number(b));
    for (const id of ids) {
      const p = personPanel(id, tracks[id], d.actions ?? []);
      panels.push(p);
      panelHost.append(p.el);
      p.update(0);
    }
  }).catch(() => {});
}
