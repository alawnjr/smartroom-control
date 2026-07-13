import type { NodeConfig } from "./types";

// Server-side node config. Set SMARTROOM_NODES as a comma-separated list of
// `id|name|host` triples; falls back to the two known nodes by IP. We default to
// raw IPs because `.local` mDNS resolution from this laptop is intermittently
// flaky (and Node's server-side resolver is worse than the browser's).
const FALLBACK = "cam1|Smartroom 1|10.61.1.169,cam2|Smartroom 2|10.61.1.206";

export const NODES: NodeConfig[] = (process.env.SMARTROOM_NODES || FALLBACK)
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean)
  .map((triple) => {
    const [id, name, host] = triple.split("|").map((p) => p.trim());
    return { id, name: name || id, host };
  })
  .filter((n) => n.id && n.host);

export function baseUrl(node: { host: string }): string {
  return `http://${node.host}:8000`;
}

// The RealSense depth page (realsense_depth_page.py) runs beside the video
// page on every node; nodes without depth cameras just return an empty list.
export function depthBaseUrl(node: { host: string }): string {
  return `http://${node.host}:8001`;
}
