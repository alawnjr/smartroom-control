// Analytics tab: session grid of per-clip cards. Each card has its OWN model
// picker (large / pose / actions NTU / actions HMDB / geometric — geometric is
// the classifier-independent jump view, treated as just another model), and
// located clips get a top-down room-map side-card next to the video card.
//
// Re-renders are gated on a JSON hash of the /api/saved listing so a poll that
// changes nothing doesn't tear down playing videos.

import { getJSON, post, h, fileUrl, groupSessions, analyzingCount, validatingCount, clipAnalyzing } from "./api.js";
import { sessionMapCard } from "./room-map.js";
import { openDrawer } from "./drawer.js";
import { parseMp4 } from "./mp4demux.js";

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

// Preview frame for a clip (1s in), extracted+cached by the v1 frame endpoint;
// clip= picks THIS card's video (a recording holds several cameras).
function posterUrl(v) {
  return `/api/v1/recordings/${encodeURIComponent(v.day)}/${encodeURIComponent(v.rec)}/${encodeURIComponent(v.node)}/frame?t=1&w=640&clip=${encodeURIComponent(v.file)}`;
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

// Dropped-frame warning: holes in the stream mean the synced player holds a
// stale frame there — the visible "one camera lags" desync. Flag clips that
// lost more than ~2% of their frames.
function dropsChip(v) {
  const dropped = v.framesDropped ?? 0;
  if (!dropped || !v.fps || !v.nominalFps || v.fps >= v.nominalFps * 0.98) return null;
  return h("span", {
    class: "chip chip-bad",
    title: `${dropped} frames lost in the capture pipeline — expect this camera to trail the others during motion`,
  }, `⚠ ${v.fps}fps · ${dropped} dropped`);
}

// ---------- synced playback drivers ----------
// Two ways to put a scheduled frame on screen. CanvasDriver (preferred) uses
// WebCodecs: the mp4 is demuxed + decoded in JS and frames are PAINTED onto a
// canvas by the master clock itself, so both cameras hit the screen in the
// same tick — <video> seek/present latency (the last unverifiable desync
// source) is out of the loop entirely. SeekDriver is the old currentTime
// stepping, kept as the fallback when WebCodecs can't handle a clip.

// Latest frame whose timestamp is <= t (binary search), -1 if none yet.
const dueIdx = (times, t) => {
  let lo = 0, hi = times.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
};

const parseCsv = (text) => {
  // the Pi writes CRLF (python csv.writer default) — trim each header cell or
  // the last column reads as "hw_timestamp_ms\r" and never matches
  const lines = text.trim().split("\n");
  const cols = lines[0].split(",").map((s) => s.trim());
  const iSec = cols.indexOf("timestamp_seconds");
  const iHw = cols.indexOf("hw_timestamp_ms");
  const sec = [], hw = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(",");
    if (iSec >= 0) sec.push(Number(c[iSec]));
    if (iHw >= 0) hw.push(Number(c[iHw]));
  }
  return { sec, hw: iHw >= 0 && hw.some((x) => x > 0) ? hw : null };
};

const ensureMeta = (el) => new Promise((res) => {
  if (el.readyState >= 1) return res();
  el.preload = "metadata";
  el.addEventListener("loadedmetadata", () => res(), { once: true });
  el.addEventListener("error", () => res(), { once: true });
  el.load();
});

// #debug HUD: a small readout on the card showing which frame is on screen
// and how far it trails the master clock.
const hudTag = (el) => {
  if (!location.hash.includes("debug")) return null;
  const wrap = el.closest(".vwrap") || el.parentElement;
  const tag = h("div", { style: "position:absolute;left:6px;bottom:6px;background:#000c;color:#3f6;" +
    "font:12px/1.4 monospace;padding:2px 7px;border-radius:6px;z-index:5;pointer-events:none" }, "sync: —");
  wrap.appendChild(tag);
  return tag;
};

class CanvasDriver {
  static async create(el, times) {
    const src = el.currentSrc || el.src;
    const res = await fetch(src);
    if (!res.ok) throw new Error(`fetch ${res.status}`);
    const buf = await res.arrayBuffer();
    const { codec, description, samples } = parseMp4(buf);
    if (!samples.length) throw new Error("no samples");
    const { supported } = await VideoDecoder.isConfigSupported({ codec, description });
    if (!supported) throw new Error(`codec unsupported: ${codec}`);
    return new CanvasDriver(el, times, buf, { codec, description }, samples);
  }

  constructor(el, times, buf, config, samples) {
    this.el = el;
    this.times = times;
    this.src = el.currentSrc || el.src;
    this.buf = buf;
    // optimizeForLatency: emit each decoded frame immediately instead of
    // buffering a reorder window — we schedule presentation ourselves
    this.config = { ...config, optimizeForLatency: true };
    this.samples = samples;
    // Sample -> CSV row via each frame's pts SLOT, not its ordinal: the Pi's
    // hw encoder occasionally drops a frame mid-encode but leaves its time
    // slot in the container, so slot index is what matches the CSV.
    const deltas = samples.slice(1).map((s, i) => s.pts - samples[i].pts).sort((a, b) => a - b);
    const step = deltas.length ? Math.max(deltas[deltas.length >> 1], 1e-3) : 1 / 30;
    const p0 = samples[0].pts;
    this.sched = samples.map((s) =>
      times[Math.min(times.length - 1, Math.max(0, Math.round((s.pts - p0) / step)))]);

    const wrap = el.closest(".vwrap") || el.parentElement;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none";
    wrap.appendChild(this.canvas);
    this.canvas.width = 0; // sized to the first decoded frame (default is 300x150)
    this.ctx = this.canvas.getContext("2d");
    this.hud = hudTag(el);

    this.ready = new Map(); // sample idx -> decoded VideoFrame awaiting its turn
    this.want = -1;         // newest due sample per the master clock
    this.wantT = 0;
    this.shown = -1;
    this.fed = 0;           // next sample to hand the decoder
    this.outIdx = 0;        // sample idx of the decoder's next output
    this.dead = false;
    this.decoder = new VideoDecoder({
      output: (frame) => this._onFrame(frame),
      error: () => { this.dead = true; },
    });
    this.decoder.configure(this.config);
  }

  showAt(t) {
    if (this.dead) return;
    const target = dueIdx(this.sched, t);
    if (target < 0) return;
    this.want = target;
    this.wantT = t;
    if (target === this.shown) return;
    this._ensurePipeline(target);
    this._feed();
    this._present();
  }

  scrub(t) { this.shown = -2; this.showAt(t); }

  // Restart decode at a keyframe when the target is behind the pipeline
  // (scrub back) or far ahead of it (big forward jump).
  _ensurePipeline(target) {
    if (this.decoder.state === "closed") { this.dead = true; return; }
    const canProduce = target >= this.outIdx || this.ready.has(target);
    if (canProduce && this.decoder.state === "configured" && target <= this.fed + 300) return;
    for (const f of this.ready.values()) f.close();
    this.ready.clear();
    if (this.decoder.state !== "unconfigured") this.decoder.reset();
    this.decoder.configure(this.config);
    let k = Math.min(target, this.samples.length - 1);
    while (k > 0 && !this.samples[k].key) k--;
    this.fed = k;
    this.outIdx = k;
  }

  _feed() {
    const PRE = 8, MAXQ = 16; // decode a touch ahead; cap in-flight work
    const stopAt = Math.min(this.want + PRE, this.samples.length - 1);
    while (this.fed <= stopAt && this.decoder.state === "configured" &&
           this.decoder.decodeQueueSize < MAXQ) {
      const s = this.samples[this.fed];
      this.decoder.decode(new EncodedVideoChunk({
        type: s.key ? "key" : "delta",
        timestamp: Math.round(s.pts * 1e6),
        data: new Uint8Array(this.buf, s.offset, s.size),
      }));
      this.fed++;
    }
  }

  _onFrame(frame) {
    const idx = this.outIdx++;
    if (idx < this.shown) frame.close(); // something newer is already on screen
    else this.ready.set(idx, frame);
    this._present();
    this._feed();
  }

  // Paint the newest decoded frame that is due; drop the ones it obsoletes.
  _present() {
    let best = -1;
    for (const k of this.ready.keys()) if (k <= this.want && k > best) best = k;
    if (best < 0 || best === this.shown) return;
    const frame = this.ready.get(best);
    this.ready.delete(best);
    for (const [k, f] of this.ready) {
      if (k < best) { f.close(); this.ready.delete(k); }
    }
    if (!this.canvas.width) {
      this.canvas.width = frame.displayWidth;
      this.canvas.height = frame.displayHeight;
    }
    this.ctx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
    frame.close();
    this.canvas.style.display = "block";
    this.shown = best;
    if (this.hud) {
      const lag = (this.wantT - this.sched[best]) * 1000;
      this.hud.textContent = `canvas f${best}/${this.samples.length} lag ${lag.toFixed(0)}ms`;
    }
  }

  destroy() {
    for (const f of this.ready.values()) f.close();
    this.ready.clear();
    try { this.decoder.close(); } catch { /* already closed */ }
    this.canvas.remove();
    if (this.hud) this.hud.remove();
  }
}

class SeekDriver {
  constructor(el, times) {
    this.el = el;
    this.times = times;
    this.src = el.currentSrc || el.src;
    this.shown = -1;
    this.lastT = null;
    // when a slow seek finishes, immediately chase the newest due frame
    // instead of waiting for the next tick
    this.onSeeked = () => { if (this.lastT != null) this.showAt(this.lastT); };
    el.addEventListener("seeked", this.onSeeked);
    this.hud = hudTag(el);
    if (this.hud && el.requestVideoFrameCallback) {
      const lags = [];
      const loop = (_, meta) => {
        const n = this.times.length, dur = el.duration || 1;
        const j = Math.min(n - 1, Math.max(0, Math.round((meta.mediaTime / dur) * n - 0.4)));
        if (this.lastT != null && n) {
          const lag = (this.lastT - this.times[j]) * 1000;
          lags.push(lag);
          if (lags.length > 30) lags.shift();
          const med = [...lags].sort((a, b) => a - b)[lags.length >> 1];
          this.hud.textContent = `video f${j}/${n} lag ${lag.toFixed(0)}ms med ${med.toFixed(0)}ms`;
        }
        el.requestVideoFrameCallback(loop);
      };
      el.requestVideoFrameCallback(loop);
    }
  }

  showAt(t) {
    this.lastT = t;
    if (!this.times.length || this.el.seeking) return; // seek in flight — seeked handler catches up
    const idx = dueIdx(this.times, t);
    if (idx < 0 || idx === this.shown) return;
    this.shown = idx;
    const n = this.times.length;
    const dur = this.el.duration || 0;
    // frame idx sits at idx/fps in the (measured-rate) container; +0.4
    // frame keeps the seek safely inside the frame's display interval
    try { this.el.currentTime = Math.min(((idx + 0.4) * dur) / n, dur); } catch { /* not seekable yet */ }
  }

  scrub(t) { this.shown = -2; this.showAt(t); }

  destroy() {
    this.el.removeEventListener("seeked", this.onSeeked);
    if (this.hud) this.hud.remove();
  }
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
  const video = h("video", { preload: "none", poster: posterUrl(v), src: fileUrl(v.relPath) });
  video.dataset.off = String(v._syncOff ?? 0);
  video.dataset.hwoff = String(v.hwOffsetMs ?? 0);
  video.dataset.csv = v.relPath.replace(/\.mp4$/, "_timestamps.csv");
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
    // while this model's analysis is pending/failed. No per-video controls —
    // the session header's Play all drives every camera together.
    const video = h("video", { preload: "none", poster: posterUrl(v), src: src() });
    video.dataset.off = String(v._syncOff ?? 0);
    video.dataset.hwoff = String(v.hwOffsetMs ?? 0);
    video.dataset.csv = v.relPath.replace(/\.mp4$/, "_timestamps.csv");
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
      h("label", { class: "who", title: "Select for validation" }, checkbox, ` ${nameFor(v.node)} `, h("span", { class: "take" }, `· ${take}`)),
      h("div", { class: "acts" }, dropsChip(v), validationChip(v), graphsBtn, reBtn, valBtn)),
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
    // No Cancel either — there is nothing local to cancel; runs live on the server.
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
    // Wall-clock offsets: cameras in a recording START at slightly different
    // moments (recorders spin up independently) — each video's timeline is
    // shifted by (its start - the session's earliest start), so the master
    // clock below plays true simultaneity, not frame-index alignment.
    const starts = s.clips.map((c) => c.startMs).filter(Boolean);
    const minStart = starts.length ? Math.min(...starts) : 0;
    const cards = [];
    for (const v of s.clips) {
      v._syncOff = v.startMs ? (v.startMs - minStart) / 1000 : 0;
      cards.push(analysisCard(v));
    }
    // ONE merged room map per recording: every located camera's tracks on the
    // shared tag-1 frame (color-coded per camera).
    const map = sessionMapCard(s.clips, nameFor);
    if (map) cards.push(map);
    // Clock-driven playback for the whole recording: ONE master clock, every
    // camera disciplined to it (its own wall-clock offset included). Videos
    // never free-run — a drifting one gets its rate nudged (or hard-seeked),
    // so what's on screen at any slider position is true simultaneity.
    const playBtn = h("button", { class: "tbtn tbtn-sm", title: "Play/pause every camera in this recording on one clock" }, "▶ Play");
    const slider = h("input", { class: "seek", type: "range", min: 0, max: 30, step: 0.01, value: 0,
                                title: "Seek every camera to this moment" });
    const timeLbl = h("span", { class: "cams", style: "min-width:5.5ch;text-align:right" }, "0.0s");
    // Replay speed: scales the master clock only — frames still land exactly
    // on their CSV timestamps, just on a slower/faster timeline.
    const speedSlider = h("input", { class: "seek", type: "range", min: 0.1, max: 2, step: 0.05, value: 1,
                                     style: "max-width:90px", title: "Replay speed (0.1×–2×)" });
    const speedLbl = h("span", { class: "cams", style: "min-width:4.5ch;text-align:right" }, "1.00×");
    const sessionEl = h("div", { class: "session" },
      h("div", { class: "session-hd" },
        h("input", { type: "checkbox", checked: allSel, title: "Select all cameras in this recording", onchange: () => {
          allSel ? clipPaths.forEach((r) => selected.delete(r)) : clipPaths.forEach((r) => selected.add(r));
          render(true);
        }}),
        h("span", { class: "lbl" }, s.label),
        h("span", { class: "cams" }, `${s.clips.length} cam${s.clips.length > 1 ? "s" : ""}`),
        playBtn,
        slider,
        timeLbl,
        speedSlider,
        speedLbl,
        h("a", { class: "tbtn tbtn-sm dl", href: `/api/saved/archive?path=${encodeURIComponent(`${day}/${rec}`)}`, title: "Download this whole recording folder as a .zip" }, "⤓ Download folder")),
      h("div", { class: "session-grid" }, ...cards));

    // Frame-scheduled playback: ONE master clock; each video gets a driver
    // (CanvasDriver = WebCodecs decode + canvas paint, SeekDriver = <video>
    // currentTime stepping as fallback) that shows the frame whose recorded
    // timestamp (from its *_timestamps.csv) has come due on the shared
    // session timeline. Display timing follows the ground-truth capture
    // times, cameras with different frame rates and start offsets included.
    const clock = { t: 0, anchor: 0, rate: 1, timer: null, loading: false, hwBase: null };
    const players = new Map();  // video el -> playback driver
    const vids = () => [...sessionEl.querySelectorAll("video")].map((el) => ({ el, off: Number(el.dataset.off || 0) }));

    // Fetch every camera's CSV once, put all frames on ONE timeline (hardware
    // timestamps when present, else metadata start offset + capture-relative
    // seconds), then build a driver per video. Re-entrant: videos whose source
    // changed (overlay toggle) get their driver rebuilt, others are kept.
    const loadPlayers = async () => {
      if (clock.loading) return;
      clock.loading = true;
      const entries = vids().filter(({ el }) => {
        const cur = players.get(el);
        if (!cur) return true;
        if (cur.src === (el.currentSrc || el.src)) return false;
        cur.destroy();
        players.delete(el);
        return true;
      });
      const raw = await Promise.all(entries.map(async ({ el, off }) => {
        // hwoff: measured inter-camera clock offset (timing calibration) —
        // subtracting it puts both cameras' hw timestamps on one true clock
        const hwoff = Number(el.dataset.hwoff || 0);
        try {
          const r = await fetch(fileUrl(el.dataset.csv));
          if (!r.ok) throw new Error();
          const parsed = parseCsv(await r.text());
          if (parsed.hw) parsed.hw = parsed.hw.map((ms) => ms - hwoff);
          return { el, off, ...parsed };
        } catch {
          return { el, off, sec: [], hw: null };
        }
      }));
      // the shared zero-point is fixed on first load so a later partial
      // rebuild (overlay toggle) stays on the same timeline
      const bases = raw.filter((x) => x.hw).map((x) => x.hw[0]);
      if (clock.hwBase === null) clock.hwBase = bases.length ? Math.min(...bases) : 0;
      for (const x of raw) {
        let times;
        if (x.hw) times = x.hw.map((ms) => (ms - clock.hwBase) / 1000);
        else if (x.sec.length) times = x.sec.map((t) => t + x.off);
        else {  // no CSV — synthesize a CFR schedule from the container
          await ensureMeta(x.el);
          const dur = x.el.duration || 0;
          const n = Math.max(1, Math.round(dur * 30));
          times = Array.from({ length: n }, (_, i) => x.off + (i * dur) / n);
        }
        let driver = null;
        if (window.VideoDecoder) {
          try {
            driver = await CanvasDriver.create(x.el, times);
          } catch (e) {
            console.warn("WebCodecs fallback for", x.el.dataset.csv, e);
          }
        }
        if (!driver) {
          await ensureMeta(x.el);
          driver = new SeekDriver(x.el, times);
        }
        players.set(x.el, driver);
      }
      clock.loading = false;
    };

    const sessionEnd = () => {
      let end = 1;
      for (const { times } of players.values()) if (times.length) end = Math.max(end, times[times.length - 1]);
      return end;
    };

    const stop = (atEnd) => {
      if (clock.timer) { clearInterval(clock.timer); clock.timer = null; }
      if (atEnd) clock.t = 0;
      playBtn.textContent = "▶ Play";
    };
    const tick = () => {
      clock.t = ((performance.now() - clock.anchor) / 1000) * clock.rate;
      const end = sessionEnd();
      for (const d of players.values()) d.showAt(clock.t);
      slider.max = end.toFixed(2);
      slider.value = String(Math.min(clock.t, end));
      timeLbl.textContent = `${clock.t.toFixed(1)}s`;
      if (clock.t >= end + 0.1) stop(true);
    };
    playBtn.addEventListener("click", async () => {
      if (clock.timer) { stop(false); return; }
      playBtn.textContent = "…";
      await loadPlayers();
      clock.anchor = performance.now() - (clock.t * 1000) / clock.rate;
      clock.timer = setInterval(tick, 33);
      playBtn.textContent = "⏸ Pause";
      tick();
    });
    slider.addEventListener("input", async () => {
      clock.t = Number(slider.value) || 0;
      clock.anchor = performance.now() - (clock.t * 1000) / clock.rate;
      timeLbl.textContent = `${clock.t.toFixed(1)}s`;
      if (!players.size) await loadPlayers();
      for (const d of players.values()) d.scrub(clock.t);
    });
    speedSlider.addEventListener("input", () => {
      clock.rate = Number(speedSlider.value) || 1;
      // re-anchor so the clock keeps its current position when the rate changes
      clock.anchor = performance.now() - (clock.t * 1000) / clock.rate;
      speedLbl.textContent = `${clock.rate.toFixed(2)}×`;
    });
    return sessionEl;
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
