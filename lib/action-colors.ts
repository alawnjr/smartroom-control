// Shared category colors so the analysis chips and the live bar graph match.
// A label's color = its index in the clip's `actions` list (same order the chips
// use), so a category looks identical in both places. Literal class strings (not
// constructed) so Tailwind keeps them.

// Chip style (light bg + dark text) — used by the action tags.
export const ACTION_TAG = [
  "bg-amber-200 text-amber-900",
  "bg-sky-200 text-sky-900",
  "bg-violet-200 text-violet-900",
  "bg-emerald-200 text-emerald-900",
  "bg-rose-200 text-rose-900",
];

// Bar fill — same hues, bolder for readability as a solid fill.
export const ACTION_BAR = [
  "bg-amber-400",
  "bg-sky-400",
  "bg-violet-400",
  "bg-emerald-400",
  "bg-rose-400",
];

// Runner-up classes that never became a chip (not in `actions`).
export const NEUTRAL_BAR = "bg-neutral-300";

export function tagClass(i: number) {
  return ACTION_TAG[i % ACTION_TAG.length];
}

// Bar color for a label, matched to its chip; neutral if it isn't a chip.
export function barClass(label: string, actions: string[]) {
  const i = actions.indexOf(label);
  return i >= 0 ? ACTION_BAR[i % ACTION_BAR.length] : NEUTRAL_BAR;
}
