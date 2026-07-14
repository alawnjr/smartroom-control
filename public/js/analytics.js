// Analytics tab: session grid of per-clip cards. Each card has its OWN model
// picker (large / pose / actions NTU / actions HMDB / geometric — geometric is
// the classifier-independent jump view, treated as just another model), and
// located clips get a top-down room-map side-card next to the video card.
//
// Re-renders are gated on a JSON hash of the /api/saved listing so a poll that
// changes nothing doesn't tear down playing videos.

import { getJSON, post, h, fileUrl, groupSessions, analyzingCount, validatingCount, clipAnalyzing } from "./api.js";
import { roomMapCard } from "./room-map.js";
import { openDrawer } from "./drawer.js";

const MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26n-pose", "action", "action-hmdb"];
const MODEL_LABEL = {
  yolo26n: "nano", yolo26s: "small", yolo26m: "medium", yolo26l: "large",
  "yolo26n-pose": "pose", action: "actions (NTU)", "action-hmdb": "actions (HMDB)",
};
// Per-card default: prefer the big detector, then pose, then actions.
const DEFAULT_ORDER = ["yolo26l", "yolo26n-pose", "action", "action-hmdb", "yolo26m", "yolo26s", "yolo26n"];
const GEOMETRIC = "geometric";
const isActionKey = (m) => m.startsWith("action");

// UI state that must survive re-renders.
const modelSel = new Map();     // relPath -> chosen model key (or "geometric")
const selected = new Set();     // relPaths checked for batch runs
const showValidation = new Set(); // relPaths with the checks panel open
let lastHash = "";
let listing = { videos: [] };
let nodeNames = new Map();
let mirror = { running: false, summary: null, failed: false };
let pollTimer = null;

function nameFor(nodeId) {
  return nodeNames.get(nodeId) ?? nodeId;
}

// Preview frame for a clip (1s in), extracted+cached by the v1 frame endpoint.
function posterUrl(v) {
  return `/api/v1/recordings/${encodeURIComponent(v.day)}/${encodeURIComponent(v.rec)}/${encodeURIComponent(v.node)}/frame?t=1&w=640`;
}

function cardModel(v) {
  const avail = MODEL_ORDER.filter((m) => v.detections?.[m]);
  const hasAction = avail.some(isActionKey);
  const options = hasAction ? [...avail, GEOMETRIC] : avail;
  const chosen = modelSel.get(v.relPath);
  if (chosen && options.includes(chosen)) return { options, model: chosen };
  const def = DEFAULT_ORDER.find((m) => avail.includes(m)) ?? avail[0] ?? "yolo26l";
  return { options, model: def };
}

// ---------- occupancy SVG (port of occupancy-graph.tsx) ----------
function occupancySVG(timeline, max) {
  const W = 320, H = 96, padL = 16, padB = 14;
  const top = Math.max(1, max);
  if (!timeline?.length) return h("div", { class: "empty", style: "height:96px;border-radius:10px;background:var(--background);display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--muted)" }, "no timeline");
  const innerW = W - padL, innerH = H - padB, n = timeline.length;
  const x = (i) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const y = (c) => (1 - c / top) * innerH;
  const pts = timeline.map((p, i) => `${x(i).toFixed(1)},${y(p.count).toFixed(1)}`).join(" ");
  let grid = "";
  for (let g = 0; g <= top; g++) {
    grid += `<line x1="${padL}" x2="${W}" y1="${y(g)}" y2="${y(g)}" stroke="rgb(0 0 0 / .06)"/><text x="0" y="${y(g) + 3}" font-size="9" fill="rgb(0 0 0 / .35)">${g}</text>`;
  }
  const div = document.createElement("div");
  div.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%" preserveAspectRatio="none">
    ${grid}
    <polygon points="${padL},${innerH} ${pts} ${W},${innerH}" fill="rgb(16 185 129 / .16)"/>
    <polyline points="${pts}" fill="none" stroke="rgb(16 185 129)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <text x="${padL}" y="${H - 2}" font-size="9" fill="rgb(0 0 0 / .4)">0s</text>
    <text x="${W}" y="${H - 2}" font-size="9" fill="rgb(0 0 0 / .4)" text-anchor="end">${Math.round(timeline[n - 1].t)}s</text>
  </svg>`;
  return div;
}

// ---------- validation ----------
function validationChip(v) {
  const val = v.validation;
  if (!val) return null;
  if (val.status === "analyzing") return h("span", { class: "chip", style: "background:var(--neutral-2);color:var(--neutral-5)" }, "validating…");
  if (val.status === "error") return h("span", { class: "chip chip-bad", title: val.error ?? "" }, "validation error");
  if (val.status !== "done") return null;
  const toggle = () => {
    showValidation.has(v.relPath) ? showValidation.delete(v.relPath) : showValidation.add(v.relPath);
    render(true);
  };
  if ((val.failed ?? 0) > 0) return h("button", { class: "chip chip-bad", title: "Click for details", onclick: toggle }, `⛨ ${val.failed} failed`);
  return h("button", { class: "chip chip-ok", title: `All ${val.passed} checks passed — click for the list`, onclick: toggle }, "⛨ valid");
}

function validationPanel(v) {
  const val = v.validation;
  if (!val || val.status !== "done" || !showValidation.has(v.relPath)) return null;
  const total = (val.passed ?? 0) + (val.failed ?? 0);
  const allOk = (val.failed ?? 0) === 0;
  const checks = [...(val.checks ?? [])].sort((a, b) => Number(a.ok) - Number(b.ok));
  return h("div", { class: `vpanel ${allOk ? "ok" : "bad"}` },
    h("div", { class: "hd" }, allOk ? `all ${total} checks passed` : `${val.failed} of ${total} checks failed`),
    h("ul", {}, ...checks.map((c) =>
      h("li", { class: c.ok ? "ok" : "bad" },
        h("span", { class: "nm" }, (c.ok ? "✓ " : "✗ ") + c.name.replaceAll("_", " ")),
        h("span", { class: "dt" }, ` — ${c.detail}`)))));
}

// ---------- geometric (jump) card body ----------
function geometricBody(v) {
  const actionModel = ["action", "action-hmdb"].find((m) => v.detections?.[m]?.status === "done");
  const video = h("video", { controls: true, preload: "none", poster: posterUrl(v), src: fileUrl(v.relPath) });
  const badge = h("span", { class: "vbadge" }, "");
  badge.hidden = true;
  const vwrap = h("div", { class: "vwrap" }, video, badge);
  const list = h("div", {}, h("div", { class: "settings-note" }, "loading events…"));

  if (actionModel) {
    const path = v.relPath.replace(/\.mp4$/, `.actions.${actionModel}.json`);
    fetch(fileUrl(path, v.detections[actionModel].version)).then((r) => (r.ok ? r.json() : null)).then((data) => {
      const events = Object.entries(data?.jumps ?? {})
        .flatMap(([id, evs]) => evs.map((e) => ({ id, ...e })))
        .sort((a, b) => a.start - b.start);
      video.addEventListener("timeupdate", () => {
        const t = video.currentTime;
        const air = events.find((e) => t >= e.start && t <= e.end);
        badge.hidden = !air;
        if (air) badge.textContent = `↑ JUMP · #${air.id}`;
      });
      list.replaceChildren(
        h("div", { class: "stat-line" },
          h("span", { class: "jump-badge" }, `↑ ${events.length} jump${events.length === 1 ? "" : "s"}`)),
        ...(events.length === 0
          ? [h("div", { class: "settings-note" }, "No jumps detected in this clip.")]
          : events.map((e) =>
              h("button", { class: "jump-row", title: "Jump to this moment", onclick: () => { video.currentTime = Math.max(0, e.start - 0.4); video.play().catch(() => {}); } },
                h("span", {}, `⭑ #${e.id}`),
                h("span", { class: "t" }, `${e.start.toFixed(1)}–${e.end.toFixed(1)}s · ↑${e.peak.toFixed(2)}`)))),
      );
    }).catch(() => list.replaceChildren(h("div", { class: "settings-note" }, "no event data")));
  } else {
    list.replaceChildren(h("div", { class: "settings-note" }, "Run an action analysis first — jumps come from the pose trajectory."));
  }
  return [vwrap, h("div", { class: "foot" }, list)];
}

// ---------- one clip card ----------
function analysisCard(v) {
  const { options, model } = cardModel(v);
  const d = model === GEOMETRIC ? null : v.detections?.[model];
  const isPose = model.includes("pose");
  const isAction = isActionKey(model);
  const analyzing = clipAnalyzing(v);
  const take = v.rec.split("_").pop();

  // header
  const checkbox = h("input", { type: "checkbox", checked: selected.has(v.relPath), onchange: () => {
    selected.has(v.relPath) ? selected.delete(v.relPath) : selected.add(v.relPath);
    render(true);
  }});
  // Analysis runs on the COSMOS node (analyze-on-node.sh) — no local re-run
  // button; just show a spinner while a run is in flight.
  const reBtn = analyzing ? h("span", { class: "icobtn", title: "Analyzing…" }, h("span", { class: "spin" }, "⟳")) : null;
  const valBtn = h("button", { class: "icobtn", title: "Re-run data validation on this clip", disabled: v.validation?.status === "analyzing", onclick: async (e) => {
    e.currentTarget.disabled = true;
    await post("/api/validate", { relPath: v.relPath, force: true });
    pingSoon();
  }}, "⛨");
  const graphsBtn = isAction && d?.status === "done"
    ? h("button", { class: "icobtn", title: "Open live graphs", onclick: () => openDrawer(v, model, d, nameFor(v.node)) }, "📈 Graphs")
    : null;

  // per-card model picker (the toggles the user asked for)
  const picker = h("div", { class: "model-row" },
    h("span", { class: "sel-count" }, "model"),
    h("div", { class: "seg" }, ...options.map((m) =>
      h("button", { class: m === model ? "on" : "", onclick: () => { modelSel.set(v.relPath, m); render(true); } },
        m === GEOMETRIC ? "geometric" : (MODEL_LABEL[m] ?? m)))));

  // body
  let body;
  if (model === GEOMETRIC) {
    body = geometricBody(v);
  } else {
    const hasOverlay = Boolean(d?.hasAnnotated && d.annotatedRelPath);
    let overlay = hasOverlay;   // per-render state; the toggle just swaps src in place
    const src = () => (overlay && hasOverlay ? fileUrl(d.annotatedRelPath, d.version) : fileUrl(v.relPath));

    const vwrap = h("div", { class: "vwrap" });
    // The raw clip is always playable (analysis only adds overlays), so every
    // card shows a real video with a preview frame; a status pill sits on top
    // while this model's analysis is pending/failed.
    const video = h("video", { controls: true, preload: "none", poster: posterUrl(v), src: src() });
    vwrap.append(video);
    if (d?.status === "done") {
      const obtn = hasOverlay ? h("button", { class: "vbtn right" }, "raw") : null;
      const sync = () => {
        video.src = src();
        if (obtn) obtn.textContent = overlay ? "raw" : isPose ? "skeleton" : isAction ? "labels" : "boxes";
      };
      if (obtn) obtn.addEventListener("click", () => { overlay = !overlay; sync(); });
      sync();
      if (obtn) vwrap.append(obtn);
    } else {
      vwrap.append(h("span", { class: "vstatus" },
        d?.status === "analyzing" ? "analyzing…" : d?.status === "error" ? "analysis failed" : "not analyzed"));
    }

    const foot = h("div", { class: "foot" });
    if (d?.status === "done" && !isAction && d.timeline) {
      foot.append(occupancySVG(d.timeline, d.maxPersons ?? 0),
        h("div", { class: "stat-line" }, `peak ${d.maxPersons} · avg ${d.avgPersons} people`));
    } else if (d?.status === "done" && isAction) {
      const tags = d.actions ?? [];
      foot.append(h("div", { class: "tags" },
        ...(tags.length === 0
          ? [h("span", { class: "settings-note" }, "no actions detected")]
          : tags.slice(0, 8).map((t, i) => h("span", { class: `tag tag-${i % 5}` }, t)))));
      if (d.stride != null) {
        foot.append(h("div", { class: "settings-note" },
          `stride ${d.stride} · ${d.samplesPerClassify ?? "?"} samples/classify · ${d.poseSource === "rtmpose" ? "RTMPose" : "YOLO pose"}`));
      }
    }
    body = [vwrap, foot];
  }

  return h("div", { class: "card card-sm acard" },
    h("div", { class: "acard-hd" },
      h("label", { class: "who", title: "Select for re-analysis" }, checkbox, ` ${nameFor(v.node)} `, h("span", { class: "take" }, `· ${take}`)),
      h("div", { class: "acts" }, validationChip(v), graphsBtn, reBtn, valBtn)),
    validationPanel(v),
    picker,
    body);
}

// ---------- toolbar ----------
function toolbar() {
  const videos = listing.videos ?? [];
  const analyzing = analyzingCount(videos);
  const validating = validatingCount(videos);
  const validated = videos.filter((v) => v.validation?.status === "done");
  const validFailed = validated.filter((v) => (v.validation?.failed ?? 0) > 0).length;
  const allRelPaths = videos.map((v) => v.relPath);
  const selCount = allRelPaths.filter((r) => selected.has(r)).length;
  const batchBody = () => (selCount > 0 ? { relPaths: allRelPaths.filter((r) => selected.has(r)), force: true } : { force: false });

  const run = (url, extra) => async () => { await post(url, { ...batchBody(), ...extra }); pingSoon(); };

  const mirrorBits = [];
  if (mirror.running) mirrorBits.push(h("span", { class: "pill pill-sky" }, h("span", { class: "dot dot-pulse dot-sky" }), " mirroring…"));
  else if (mirror.failed) mirrorBits.push(h("span", { class: "pill pill-rose" }, "mirror sync failed"));
  else if (mirror.summary) mirrorBits.push(h("span", { class: "mirror-note", title: mirror.summary }, mirror.summary.replace(/^manifest:\s*/, "").split("|").slice(1).join("·").trim()));

  return [
    h("span", { class: "sel-count" }, selCount > 0 ? `${selCount} selected` : "all clips"),
    h("button", { class: "tbtn tbtn-sm", onclick: () => { allRelPaths.forEach((r) => selected.add(r)); render(true); } }, "Select all"),
    h("button", { class: "tbtn tbtn-sm", disabled: selCount === 0, onclick: () => { selected.clear(); render(true); } }, "Clear"),
    // No local "Analyze" button: analysis runs on the COSMOS node via
    // analyze-on-node.sh (analyzing counts still show progress when results
    // are pulled back / a local run was started by hand).
    analyzing > 0 ? h("span", { class: "pill" }, h("span", { class: "spin" }, "⟳"), ` Analyzing ${analyzing}…`) : null,
    h("button", { class: "tbtn", disabled: validating > 0, title: "Data-integrity checks on the selected clips, or all if none selected", onclick: run("/api/validate") },
      validating > 0 ? h("span", { class: "spin" }, "⟳") : "⛨", validating > 0 ? ` Validating ${validating}…` : selCount > 0 ? ` Validate ${selCount}` : " Validate all"),
    validated.length > 0 && validating === 0
      ? h("span", { class: `chip ${validFailed > 0 ? "chip-bad" : "chip-ok"}`, title: validFailed > 0 ? "Some clips failed validation — see the red chips on their cards" : "All validated clips passed" },
          validFailed > 0 ? `⛨ ${validFailed}/${validated.length} flagged` : `⛨ ${validated.length} valid`)
      : null,
    analyzing > 0 ? h("button", { class: "tbtn tbtn-cancel", onclick: async () => { await post("/api/detect/cancel"); pingSoon(); } }, "✕ Cancel") : null,
    h("span", { class: "grow" }),
    h("button", { class: "tbtn", title: "Pull new recordings from the Pis (analysis happens on the server: analyze-on-node.sh)",
        onclick: async (e) => { e.target.disabled = true; try { await post("/api/save-all"); } finally { e.target.disabled = false; } pingSoon(); } },
      "⤓ Beam to laptop"),
    ...mirrorBits,
    h("button", { class: "tbtn", disabled: mirror.running, title: "Upload new recordings + inference to the public Vercel mirror", onclick: async () => { await post("/api/mirror"); pollMirror(); } }, "☁ Sync to mirror"),
  ].filter(Boolean);
}

// ---------- render ----------
function render(force = false) {
  const hash = JSON.stringify(listing) + JSON.stringify([...selected]) + JSON.stringify([...showValidation]) + JSON.stringify([...modelSel]) + JSON.stringify(mirror);
  if (!force && hash === lastHash) return;
  lastHash = hash;

  document.getElementById("an-toolbar").replaceChildren(...toolbar());
  const host = document.getElementById("an-sessions");
  const videos = listing.videos ?? [];
  if (videos.length === 0) {
    host.replaceChildren(h("p", { class: "empty-note" }, "No clips to analyze yet — record on a node's page, then “Beam to laptop” above."));
    return;
  }
  const sessions = groupSessions(videos);
  host.replaceChildren(...sessions.map((s) => {
    const clipPaths = s.clips.map((c) => c.relPath);
    const allSel = clipPaths.every((r) => selected.has(r));
    const { day, rec } = s.clips[0];
    const cards = [];
    for (const v of s.clips) {
      cards.push(analysisCard(v));
      const map = roomMapCard(v, nameFor(v.node)); // side-card when the clip is located
      if (map) cards.push(map);
    }
    return h("div", { class: "session" },
      h("div", { class: "session-hd" },
        h("input", { type: "checkbox", checked: allSel, title: "Select all cameras in this recording", onchange: () => {
          allSel ? clipPaths.forEach((r) => selected.delete(r)) : clipPaths.forEach((r) => selected.add(r));
          render(true);
        }}),
        h("span", { class: "lbl" }, s.label),
        h("span", { class: "cams" }, `${s.clips.length} cam${s.clips.length > 1 ? "s" : ""}`),
        h("a", { class: "tbtn tbtn-sm dl", href: `/api/saved/archive?path=${encodeURIComponent(`${day}/${rec}`)}`, title: "Download this whole recording folder as a .zip" }, "⤓ Download folder")),
      h("div", { class: "session-grid" }, ...cards));
  }));
}

// ---------- polling ----------
async function poll() {
  try {
    const data = await getJSON("/api/saved");
    listing = data;
    render();
  } catch { /* server briefly away — keep the last render */ }
  const busy = analyzingCount(listing.videos ?? []) > 0 || validatingCount(listing.videos ?? []) > 0;
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, busy ? 2000 : 10000);
}

// After starting a batch, detect.py takes a moment to write its "analyzing"
// marker — re-check a few times so the busy poll kicks in (port of pingSavedSoon).
function pingSoon() {
  for (const ms of [300, 1000, 2000, 4000, 7000, 11000]) setTimeout(poll, ms);
}

async function pollMirror() {
  try {
    mirror = await getJSON("/api/mirror");
  } catch { return; }
  render();
  if (mirror.running) setTimeout(pollMirror, 2000);
}

export async function initAnalytics() {
  try {
    const status = await getJSON("/api/status");
    nodeNames = new Map((status.nodes ?? []).map((n) => [n.id, n.name]));
  } catch { /* names fall back to ids */ }
  pollMirror();
  poll();
}
