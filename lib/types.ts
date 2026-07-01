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
  detections?: Record<string, DetectionSummary>; // slot 1 / in-place: shared detection + original action keys
  analyses?: Record<number, SlotAnalysis>; // extra action-analysis slots (>=2), keyed by slot number
};

// One saved analysis slot (>=2): its settings snapshot + the action sidecars it produced.
export type SlotConfig = {
  settings?: { stride?: number; samplesPerClassify?: number };
  variants?: string[];
  createdAt?: string;
  [key: string]: unknown; // per-variant { disabled: string[] }
};
export type SlotAnalysis = {
  slot: number;
  config?: SlotConfig;
  detections: Record<string, DetectionSummary>;
};

export type DetectionStatus = "analyzing" | "done" | "error" | "none";

export type DetectionTimelinePoint = { t: number; count: number };

export type DetectionSummary = {
  model: string; // e.g. yolo26n / yolo26s / yolo26m
  status: DetectionStatus;
  maxPersons?: number;
  avgPersons?: number;
  framesAnalyzed?: number;
  durationSec?: number;
  timeline?: DetectionTimelinePoint[];
  hasAnnotated: boolean;
  annotatedRelPath?: string; // recordings-relative, for /api/saved/file
  actionsRelPath?: string; // action models: recordings-relative path to .actions.<model>.json
  version?: number; // sidecar mtime (ms) — cache-buster so a re-run yields fresh URLs
  error?: string;
  // action model only:
  tracks?: number;
  actions?: string[]; // distinct action labels seen
  trackActions?: Record<string, string>; // track id -> dominant action
  jumps?: number; // geometric jump events detected (count)
};

export type SavedListing = { root: string; videos: SavedVideo[] };
