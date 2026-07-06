// A configured camera node (from SMARTROOM_NODES).
export type NodeConfig = {
  id: string;
  name: string;
  host: string; // bare host or IP — goes straight into a URL
};

// The Pi's /record/status payload.
export type PiStatus = {
  running: boolean;
  duration: number;
  elapsed: number;
  remaining: number;
};

// One node's combined liveness + status (from /api/status).
export type NodeStatus = NodeConfig & {
  online: boolean;
  status: PiStatus | null;
  error?: string;
};

export type CombinedStatus = { nodes: NodeStatus[] };

// One node's outcome from a record/cancel fan-out.
export type RecordResult = {
  id: string;
  name: string;
  ok: boolean;
  httpStatus: number | null;
  message: string;
};

export type RecordResponse = { results: RecordResult[]; allOk: boolean };

// One node's outcome from "Save All to Laptop".
export type SaveResult = {
  id: string;
  name: string;
  downloaded: number;
  skipped: number;
  failed: number;
  bytes: number;
  error?: string;
};

export type SaveResponse = { saveRoot: string; nodes: SaveResult[] };

// A video saved under recordings/ (cam1/day_*/rec_*/streams/file.mp4).
export type SavedVideo = {
  node: string; // cam1 / cam2
  day: string; // day_NN_YYYY-MM-DD
  rec: string; // rec_YYYYMMDD_NNN
  file: string; // camera_main.mp4
  relPath: string; // full path relative to the recordings root
  size: number;
  mtime: number;
  detections?: Record<string, DetectionSummary>; // keyed by model (yolo26n/s/m)
};

export type DetectionStatus = "analyzing" | "done" | "error" | "none";

export type DetectionTimelinePoint = { t: number; count: number };

export type DetectionSummary = {
  model: string; // e.g. yolo26n / yolo26s / yolo26m
  status: DetectionStatus;
  maxPersons?: number;
  avgPersons?: number;
  framesAnalyzed?: number;
  timeline?: DetectionTimelinePoint[];
  hasAnnotated: boolean;
  annotatedRelPath?: string; // recordings-relative, for /api/saved/file
  error?: string;
};

export type SavedListing = { root: string; videos: SavedVideo[] };
