// Live tab: room MJPEG tiles + the record panel. No clip gallery here — recorded
// videos live in the Analytics tab.
//
// Room cards are built once per node and then UPDATED IN PLACE: rebuilding the
// DOM each poll would tear down the MJPEG <img> and restart the stream every
// second. Only state flips (live <-> recording <-> offline) swap the tile body.

import { getJSON, post, h, fmtClock, fmtDur } from "./api.js";

const cards = new Map(); // node id -> {root, refs, state}
let duration = 30;
let statusData = null;
export let nodesConfig = []; // [{id,name,host,online}] — analytics uses names too

function streamUrl(id) {
  return `/api/stream/${encodeURIComponent(id)}`;
}

function buildCard(node, idx) {
  const refs = {};
  refs.pill = h("span", { class: "pill" });
  refs.body = h("div", {}); // placeholder / img slot
  refs.progress = h("div", {}, (refs.progressFill = h("div", { style: "width:0%" })));
  refs.progress.className = "progress";
  refs.progress.hidden = true;
  refs.tile = h("div", { class: "room-tile" }, refs.pill, refs.body, refs.progress);
  refs.state = h("span", { class: "room-state" });
  refs.dots = h("div", { class: "room-dots" });
  const root = h(
    "div",
    { class: `card card-sm room-card room-${idx % 4}` },
    (refs.bar = h("div", { class: "bar" })),
    h("div", { class: "inner" },
      refs.tile,
      h("div", { class: "room-meta" },
        h("div", {},
          h("div", { class: "room-name" }, node.name),
          h("div", { class: "room-host" }, node.host ?? "")),
        h("div", { class: "room-side" }, refs.dots, refs.state))),
  );
  return { root, refs, state: "" };
}

function setTileState(card, node) {
  const { refs } = card;
  const running = Boolean(node.online && node.status?.running);
  const state = !node.online ? "offline" : running ? "recording" : "live";

  if (state !== card.state) {
    card.state = state;
    refs.body.remove();
    if (state === "offline") {
      refs.tile.className = "room-tile grad-off stripes";
      refs.pill.className = "pill";
      refs.pill.style.cssText = "background:rgb(229 229 229/.9);color:var(--neutral-5)";
      refs.pill.textContent = "Off the grid";
      refs.body = h("div", { class: "placeholder", style: "color:var(--neutral-4)" }, "Napping");
      refs.dots.replaceChildren(...[0, 1].map(() => h("span", { class: "dot", style: "background:var(--neutral-3)" })));
      refs.state.replaceChildren(h("button", { class: "wake", onclick: () => refresh() }, "wake it up →"));
    } else if (state === "recording") {
      refs.tile.className = "room-tile grad";
      refs.pill.className = "pill pill-rose";
      refs.pill.style.cssText = "";
      refs.body = h("div", { class: "placeholder scanlines", style: "color:rgb(251 113 133/.7)" }, "recording…");
      refs.dots.replaceChildren(...[0, 1, 2].map(() => h("span", { class: "dot dot-id" })));
      refs.state.textContent = "on camera";
    } else {
      refs.tile.className = "room-tile grad";
      refs.pill.className = "pill";
      refs.pill.style.cssText = "background:rgb(255 255 255/.8);color:var(--emerald-tx,#047857)";
      refs.pill.replaceChildren(h("span", { class: "dot dot-emerald" }), " Live");
      const img = h("img", { class: "scanlines", alt: `${node.name} live` });
      let retryKey = 0;
      img.addEventListener("error", () => setTimeout(() => { img.src = `${streamUrl(node.id)}?k=${++retryKey}`; }, 1500));
      img.src = streamUrl(node.id);
      refs.body = img;
      refs.dots.replaceChildren(...[0, 1, 2].map(() => h("span", { class: "dot dot-id" })));
      refs.state.textContent = "live";
    }
    refs.tile.insertBefore(refs.body, refs.progress);
  }

  if (state === "recording") {
    const st = node.status;
    refs.pill.replaceChildren(h("span", { class: "dot dot-pulse dot-rose" }), ` Rolling · ${fmtClock(st.remaining)}`);
    refs.progress.hidden = false;
    const pct = st.duration ? Math.min(100, (st.elapsed / st.duration) * 100) : 0;
    refs.progressFill.style.width = `${pct}%`;
  } else {
    refs.progress.hidden = true;
  }
}

async function refresh() {
  let data;
  try {
    data = await getJSON("/api/status");
  } catch {
    return;
  }
  statusData = data;
  nodesConfig = data.nodes ?? [];
  const grid = document.getElementById("rooms");
  data.nodes.forEach((node, idx) => {
    let card = cards.get(node.id);
    if (!card) {
      card = buildCard(node, idx);
      cards.set(node.id, card);
      grid.append(card.root);
    }
    setTileState(card, node);
  });

  const live = data.nodes.filter((n) => n.online).length;
  const rolling = data.nodes.filter((n) => n.online && n.status?.running).length;
  document.getElementById("pill-live").textContent = `${live} rooms live`;
  const rp = document.getElementById("pill-rolling");
  rp.hidden = rolling === 0;
  document.getElementById("rolling-n").textContent = rolling;
  document.getElementById("btn-stop").disabled = rolling === 0;
}

async function refreshStats() {
  let data;
  try {
    data = await getJSON("/api/analyze-stats");
  } catch {
    return;
  }
  const runs = data.runs ?? [];
  const host = document.getElementById("analyze-stats");
  if (!runs.length) return void (host.hidden = true);
  host.hidden = false;
  host.replaceChildren(
    h("div", { class: "hd" }, "Last analysis runs"),
    ...runs.map((r) =>
      h("div", { class: "row" },
        h("b", {}, r.label),
        h("span", {},
          `${r.processed} clip${r.processed === 1 ? "" : "s"} · ${fmtDur(r.elapsedSec)}` +
          (r.perClipSec ? ` · ${fmtDur(r.perClipSec)}/clip` : "") +
          (r.errors ? ` · ${r.errors} err` : "")))),
  );
}

export function initLive() {
  const durVal = document.getElementById("dur-val");
  const step = (d) => {
    duration = Math.max(5, Math.min(3600, duration + d));
    durVal.textContent = `${duration}s`;
  };
  document.getElementById("dur-minus").addEventListener("click", () => step(-5));
  document.getElementById("dur-plus").addEventListener("click", () => step(5));
  document.getElementById("btn-record").addEventListener("click", async (e) => {
    e.target.disabled = true;
    await post("/api/record", { duration });
    e.target.disabled = false;
    refresh();
  });
  document.getElementById("btn-stop").addEventListener("click", async () => {
    await post("/api/cancel");
    refresh();
  });
  document.getElementById("btn-beam").addEventListener("click", async (e) => {
    e.target.disabled = true;
    e.target.textContent = "beaming…";
    await post("/api/save-all");
    post("/api/detect"); // analyze whatever is new
    e.target.disabled = false;
    e.target.textContent = "⤓ Beam to laptop";
  });

  const clock = document.getElementById("clock");
  const tick = () => (clock.textContent = new Date().toLocaleTimeString());
  tick();
  setInterval(tick, 1000);

  refresh();
  refreshStats();
  setInterval(refresh, 1000);
  setInterval(refreshStats, 8000);
}
