// Classes tab: browse the full label set each action model can emit and toggle
// classes on/off. Disabled classes are masked at inference (detect/action.py);
// "apply" forces a re-run so the change takes effect. Port of
// action-classes-page.tsx — WITHOUT the old stride / samples-per-classify
// controls (those settings are pinned in action-classes.json now).

import { getJSON, post, h } from "./api.js";

let config = {};   // { [variant]: {disabled: []}, settings: {...} }
let datasets = []; // label catalogs from the API (_datasets)
let query = "";

function datasetCard(ds) {
  const disabled = new Set(config[ds.key]?.disabled ?? []);
  const q = query.trim().toLowerCase();
  const rows = ds.classes.map((name, i) => ({ name, i })).filter(({ name }) => !q || name.toLowerCase().includes(q));
  const enabledCount = ds.classes.length - disabled.size;
  const variant = ds.key === "action-hmdb" ? "hmdb" : "ntu";

  const save = (next) => {
    config = { ...config, [ds.key]: { disabled: next } };
    render();
    post("/api/action-classes", { variant: ds.key, disabled: next });
  };
  const toggle = (name) => {
    const next = new Set(config[ds.key]?.disabled ?? []);
    next.has(name) ? next.delete(name) : next.add(name);
    save([...next]);
  };

  return h("div", { class: "card ds-card" },
    h("div", { class: "ds-hd" },
      h("div", {},
        h("div", { class: "nm" }, ds.label),
        h("div", { class: "meta" }, h("span", { class: "mono" }, ds.model), ` · ${ds.dataset}`)),
      h("div", { class: "ds-actions" },
        h("span", { class: "count-pill" }, `${enabledCount} / ${ds.classes.length} on`),
        h("button", { class: "tbtn tbtn-sm", onclick: () => save([]) }, "all on"),
        h("button", { class: "tbtn tbtn-sm", onclick: () => save([...ds.classes]) }, "all off"),
        // Analysis is server-only: toggles save instantly to action-classes.json
        // (which push() ships to the node) and take effect on the next
        // analyze-on-node.sh run — nothing to launch from the laptop.
        h("span", { class: "meta", title: "Toggles are saved immediately; re-run analyze-on-node.sh (FORCE=1 for actions) to apply them" },
          "applies on next server run"))),
    h("p", { class: "ds-blurb" }, ds.blurb),
    rows.length === 0
      ? h("div", { class: "empty-note" }, `No classes match “${query}”.`)
      : h("div", { class: "classes-grid" }, ...rows.map(({ name, i }) => {
          const on = !disabled.has(name);
          return h("button", { class: `cls ${on ? "" : "off"}`, title: on ? "enabled — click to disable" : "disabled — click to enable", onclick: () => toggle(name) },
            h("span", { class: "sw" }, h("i", {})),
            h("span", { class: "idx" }, String(i)),
            h("span", { class: "nm", title: name }, name));
        })));
}

function render() {
  const root = document.getElementById("classes-root");
  const search = h("input", { placeholder: "filter classes…", value: query });
  search.addEventListener("input", () => { query = search.value; render(); requestAnimationFrame(() => {
    const el = document.querySelector(".search input");
    if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); }
  }); });

  root.replaceChildren(
    h("div", { class: "classes-hd" },
      h("div", {},
        h("h2", { class: "h2", style: "margin-bottom:0" }, "Action classes"),
        h("p", { class: "sub" }, "Toggle classes off to mask them at inference — the model picks the best enabled class or falls back to idle. Re-analyze clips to apply.")),
      h("label", { class: "search" }, "🔍", search)),
    ...datasets.map(datasetCard));
}

export async function initClasses() {
  try {
    const data = await getJSON("/api/action-classes");
    datasets = data._datasets ?? [];
    delete data._datasets;
    config = data;
  } catch { /* leave empty */ }
  render();
}
