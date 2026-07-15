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
  startMs?: number; // stream's wall-clock start (epoch ms, from metadata.json) — aligns cameras on one timeline
  hwOffsetMs?: number; // measured inter-camera clock offset — subtract from this stream's hw timestamps
  fps?: number; // measured delivery rate (from metadata.json)
  nominalFps?: number; // the rate the camera was configured for
  framesDropped?: number; // frames lost in the capture pipeline (holes → visible desync)
  detections?: Record<string, DetectionSummary>; // in-place per-model detection + action results
  validation?: ValidationSummary; // data-integrity checks (detect/validate.py sidecar)
  // Lens-corrected copy (undistort.py), when this clip is calibrated + processed:
  undistortedRelPath?: string;
  undistortedVersion?: number; // file mtime — cache-buster
};

export type DetectionStatus = "analyzing" | "done" | "error" | "none";

// One clip's data-validation outcome (camera_main.validation.json).
export type ValidationSummary = {
  status: DetectionStatus;
  passed?: number;
  failed?: number;
  failedChecks?: string[]; // names of failed checks (for the chip tooltip)
  checks?: { name: string; ok: boolean; detail: string }[]; // every check with its outcome + human explanation (click-to-expand panel)
  version?: number; // sidecar mtime — cache-buster so a re-run re-renders
  error?: string;
};

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
  // settings that produced the run (action models), for the header summary:
  stride?: number;
  samplesPerClassify?: number;
  poseSource?: "yolo" | "rtmpose";
};

export type SavedListing = { root: string; videos: SavedVideo[] };
