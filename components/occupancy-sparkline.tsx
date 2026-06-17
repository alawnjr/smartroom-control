import type { DetectionTimelinePoint } from "@/lib/types";

// Tiny dependency-free occupancy sparkline (person count over time).
export function OccupancySparkline({
  timeline,
  max,
}: {
  timeline: DetectionTimelinePoint[];
  max: number;
}) {
  const W = 100;
  const H = 24;
  if (!timeline || timeline.length === 0) {
    return <div className="h-6 w-full rounded bg-neutral-800/40" />;
  }
  const top = Math.max(1, max);
  const n = timeline.length;
  const x = (i: number) => (n === 1 ? 0 : (i / (n - 1)) * W);
  const y = (c: number) => H - (c / top) * H;
  const pts = timeline.map((p, i) => `${x(i).toFixed(1)},${y(p.count).toFixed(1)}`).join(" ");
  const area = `0,${H} ${pts} ${W},${H}`;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-6 w-full"
      aria-label={`occupancy timeline, peak ${max}`}
    >
      <polygon points={area} fill="rgb(16 185 129 / 0.15)" />
      <polyline
        points={pts}
        fill="none"
        stroke="rgb(16 185 129)"
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
