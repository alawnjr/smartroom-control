// Entry point: tab switching + lazy-init of each tab's module.
//
// The Live tab (room streams + record panel) is gone — live view and recording
// happen on the nodes' own pages (http://smartroom<N>.local:8000, which has a
// "Record ALL nodes" button); this dashboard is analysis-only.

import { initAnalytics } from "./analytics.js";
import { initClasses } from "./classes.js";

const inited = new Set();
const init = { analytics: initAnalytics, classes: initClasses };

function show(tab) {
  for (const t of ["analytics", "classes"]) {
    document.getElementById(`tab-${t}`).hidden = t !== tab;
  }
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  if (!inited.has(tab)) {
    inited.add(tab);
    init[tab]();
  }
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  location.hash = b.dataset.tab === "analytics" ? "" : b.dataset.tab;
  show(b.dataset.tab);
}));

// Deep-linkable tabs: /#classes.
const fromHash = () => (location.hash.slice(1) === "classes" ? "classes" : "analytics");
window.addEventListener("hashchange", () => show(fromHash()));
show(fromHash());

const clock = document.getElementById("clock");
const tick = () => (clock.textContent = new Date().toLocaleTimeString());
tick();
setInterval(tick, 1000);

// Stale-tab guard: this SPA lives for days while new recordings poll in, so a
// long-open tab can run week-old player code against fresh data. Watch the
// module's ETag and reload once it changes (on focus + every 10 min).
let jsTag = null;
async function checkFresh() {
  try {
    const r = await fetch("/js/analytics.js", { method: "HEAD", cache: "no-store" });
    const tag = r.headers.get("etag") || r.headers.get("last-modified");
    if (jsTag === null) jsTag = tag;
    else if (tag && tag !== jsTag) location.reload();
  } catch {
    /* server briefly down — retry on the next trigger */
  }
}
checkFresh();
window.addEventListener("focus", checkFresh);
setInterval(checkFresh, 10 * 60 * 1000);
