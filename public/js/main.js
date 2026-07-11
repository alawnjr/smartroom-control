// Entry point: tab switching + lazy-init of each tab's module.

import { initLive } from "./live.js";
import { initAnalytics } from "./analytics.js";
import { initClasses } from "./classes.js";

const inited = new Set();
const init = { live: initLive, analytics: initAnalytics, classes: initClasses };

function show(tab) {
  for (const t of ["live", "analytics", "classes"]) {
    document.getElementById(`tab-${t}`).hidden = t !== tab;
  }
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  if (!inited.has(tab)) {
    inited.add(tab);
    init[tab]();
  }
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  location.hash = b.dataset.tab === "live" ? "" : b.dataset.tab;
  show(b.dataset.tab);
}));

// Deep-linkable tabs: /#analytics, /#classes.
const fromHash = () => (["analytics", "classes"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "live");
window.addEventListener("hashchange", () => show(fromHash()));
show(fromHash());
