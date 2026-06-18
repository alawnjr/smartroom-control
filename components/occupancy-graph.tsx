import type { DetectionTimelinePoint } from "@/lib/types";

// People-over-time area chart (inline SVG, no deps). Faint integer gridlines,
// emerald area + line, peak labeled.
export function OccupancyGraph({ timeline, max }: { timeline: DetectionTimelinePoint[]; max: number }) {
  const W = 320;
  const H = 96;
  const padL = 16;
  const padB = 14;
  const top = Math.max(1, max);

  if (!timeline || timeline.length === 0) {
    return <div className="flex h-[96px] items-center justify-center rounded-lg bg-background text-xs text-muted">no timeline</div>;
  }

  const innerW = W - padL;
  const innerH = H - padB;
  const n = timeline.length;
  const x = (i: number) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const y = (c: number) => (1 - c / top) * innerH;
  const pts = timeline.map((p, i) => `${x(i).toFixed(1)},${y(p.count).toFixed(1)}`).join(" ");
  const lastT = timeline[n - 1].t;

  // gridlines at each integer person-count
  const lines = [];
  for (let g = 0; g <= top; g++) lines.push(g);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" role="img" aria-label={`occupancy over time, peak ${max}`}>
      {lines.map((g) => (
        <g key={g}>
          <line x1={padL} x2={W} y1={y(g)} y2={y(g)} stroke="rgb(0 0 0 / 0.06)" strokeWidth={1} />
          <text x={0} y={y(g) + 3} fontSize={9} fill="rgb(0 0 0 / 0.35)">{g}</text>
        </g>
      ))}
      <polygon points={`${padL},${innerH} ${pts} ${W},${innerH}`} fill="rgb(16 185 129 / 0.16)" />
      <polyline points={pts} fill="none" stroke="rgb(16 185 129)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      <text x={padL} y={H - 2} fontSize={9} fill="rgb(0 0 0 / 0.4)">0s</text>
      <text x={W} y={H - 2} fontSize={9} fill="rgb(0 0 0 / 0.4)" textAnchor="end">{Math.round(lastT)}s</text>
    </svg>
  );
}
