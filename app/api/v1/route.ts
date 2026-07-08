import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Self-describing index for the LAN machine API (read-only). Point another
// device at http://<this-host>:4000/api/v1 and it can discover everything.
export async function GET(req: NextRequest) {
  const o = req.nextUrl.origin;
  return NextResponse.json(
    {
      name: "smartroom-control API",
      version: 1,
      readOnly: true,
      docs: `${o}/api/v1/docs`,
      endpoints: {
        recordings: {
          url: `${o}/api/v1/recordings`,
          description:
            "All recording sessions (newest first) with per-camera analysis status, validation, calibration flag, and per-resource URLs.",
        },
        inference: {
          url: `${o}/api/v1/recordings/{day}/{rec}/{cam}/inference/{model}`,
          description:
            "Every inference sidecar for one clip+model merged into one JSON: detections summary/timeline, action timelines, per-person segments+keypoints, centroids, raw pose keypoints; plus the camera calibration when present.",
          models: ["yolo26l", "yolo26n-pose", "action", "action-hmdb"],
        },
        frame: {
          url: `${o}/api/v1/recordings/{day}/{rec}/{cam}/frame?t=2.5&w=640&q=80&video=raw`,
          description:
            "One compressed JPEG frame at time t seconds. w = output width px (64-1920, default 640), q = JPEG quality 1-100 (default 80), video = raw | undistorted | annotated.<model>.",
        },
        video: {
          url: `${o}/api/v1/recordings/{day}/{rec}/{cam}/video?variant=raw`,
          description:
            "Whole video (307 redirect to a range-capable file endpoint). variant = raw | undistorted | annotated.<model>.",
        },
      },
      example: `${o}/api/v1/recordings`,
    },
    { headers: { "access-control-allow-origin": "*", "cache-control": "no-store" } }
  );
}
