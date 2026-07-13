import { NextResponse } from "next/server";

import { NODES, depthBaseUrl } from "@/lib/nodes";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface DepthDeviceStatus {
  running?: boolean;
  starting?: boolean;
  error?: string | null;
  info?: { profile?: string; usb?: string; firmware?: string };
}

interface DepthDevice {
  name: string;
  serial: string;
  usb: string;
  status?: DepthDeviceStatus;
}

// Fan out to every node's RealSense page (:8001/devices). A node without the
// depth page (or offline) maps to an empty device list rather than an error,
// so rooms with only a regular webcam don't produce noise.
export async function GET() {
  const nodes = await Promise.all(
    NODES.map(async (n) => {
      try {
        const res = await fetch(`${depthBaseUrl(n)}/devices`, {
          signal: AbortSignal.timeout(2000),
          cache: "no-store",
        });
        if (!res.ok) return { ...n, online: false, devices: [] as DepthDevice[] };
        const body = (await res.json()) as { devices?: DepthDevice[] };
        return { ...n, online: true, devices: body.devices ?? [] };
      } catch {
        return { ...n, online: false, devices: [] as DepthDevice[] };
      }
    })
  );
  return NextResponse.json({ nodes }, { headers: { "cache-control": "no-store" } });
}
