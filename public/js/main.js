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
