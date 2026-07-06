// Action label lists, index-aligned with each model head (kept in sync with the
// canonical lists in detect/action.py). NTU = 2D ST-GCN++ on NTU-RGB+D 60;
// HMDB = PoseC3D on HMDB51.
export const NTU60: string[] = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up", "clapping", "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses", "take off glasses", "put on a hat", "take off a hat", "cheer up", "hand waving", "kick something", "reach into pocket", "hopping", "jump up", "phone call", "play with phone", "type on keyboard", "point to something", "take a selfie", "check time", "rub two hands", "nod head/bow", "shake head", "wipe face", "salute", "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down", "headache", "chest pain", "back pain", "neck pain", "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing", "pat on back", "point finger", "hugging", "give object", "touch pocket", "handshake", "walk towards", "walk apart"];

export const HMDB51: string[] = ["brush hair", "cartwheel", "catch", "chew", "clap", "climb", "climb stairs", "dive", "draw sword", "dribble", "drink", "eat", "fall floor", "fencing", "flic flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride bike", "ride horse", "run", "shake hands", "shoot ball", "shoot bow", "shoot gun", "sit", "situp", "smile", "smoke", "somersault", "stand", "swing baseball", "sword", "sword exercise", "talk", "throw", "turn", "walk", "wave"];

export const DATASETS = [
  {
    key: "action",
    label: "actions (NTU)",
    model: "2D ST-GCN++",
    dataset: "NTU-RGB+D 60",
    blurb: "60 mostly office/daily gestures. Lightweight skeleton model; no plain walk/run.",
    classes: NTU60,
  },
  {
    key: "action-hmdb",
    label: "actions (HMDB)",
    model: "PoseC3D",
    dataset: "HMDB51",
    blurb: "51 everyday + sport actions (adds walk/run). Heavier per-inference.",
    classes: HMDB51,
  },
] as const;
