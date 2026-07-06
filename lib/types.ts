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
