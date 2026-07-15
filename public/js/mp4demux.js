// Minimal MP4 (ISO-BMFF) demuxer for the synced player's WebCodecs path.
//
// Scope: the plain, non-fragmented single-video-track files this project
// produces (ffmpeg mp4 mux, H.264). Parses the sample tables and returns
// everything VideoDecoder needs: the codec string + avcC description and one
// {offset, size, pts, key} entry per sample. Not a general-purpose demuxer.

class Box {
  constructor(view, start, size, type) {
    this.view = view;
    this.start = start; // offset of the box header
    this.size = size;
    this.type = type;
  }
  get bodyStart() {
    return this.start + 8;
  }
  *children() {
    let p = this.bodyStart;
    const end = this.start + this.size;
    while (p + 8 <= end) {
      let size = this.view.getUint32(p);
      const type = String.fromCharCode(
        this.view.getUint8(p + 4), this.view.getUint8(p + 5),
        this.view.getUint8(p + 6), this.view.getUint8(p + 7));
      if (size === 1) size = Number(this.view.getBigUint64(p + 8)); // largesize
      if (size < 8 || p + size > end) return;
      yield new Box(this.view, p, size, type);
      p += size;
    }
  }
  find(type) {
    for (const c of this.children()) if (c.type === type) return c;
    return null;
  }
  path(...types) {
    let box = this;
    for (const t of types) {
      box = box.find(t);
      if (!box) return null;
    }
    return box;
  }
}

const u32s = (box, skip, n) => {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = box.view.getUint32(box.bodyStart + skip + i * 4);
  return out;
};

// The video trak: the one whose hdlr handler_type is 'vide'.
function videoTrak(moov) {
  for (const trak of moov.children()) {
    if (trak.type !== "trak") continue;
    const hdlr = trak.path("mdia", "hdlr");
    if (!hdlr) continue;
    const v = hdlr.view;
    const h = String.fromCharCode(v.getUint8(hdlr.bodyStart + 8), v.getUint8(hdlr.bodyStart + 9),
      v.getUint8(hdlr.bodyStart + 10), v.getUint8(hdlr.bodyStart + 11));
    if (h === "vide") return trak;
  }
  return null;
}

// avc1 sample entry -> codec string ("avc1.PPCCLL") + raw avcC payload
// (the AVCDecoderConfigurationRecord VideoDecoder wants as `description`).
function avcConfig(stsd, buf) {
  // stsd body: version/flags (4) + entry_count (4), then the first sample entry
  const entry = new Box(stsd.view, stsd.bodyStart + 8,
    stsd.view.getUint32(stsd.bodyStart + 8), "entry");
  entry.type = String.fromCharCode(stsd.view.getUint8(entry.start + 4), stsd.view.getUint8(entry.start + 5),
    stsd.view.getUint8(entry.start + 6), stsd.view.getUint8(entry.start + 7));
  if (entry.type !== "avc1" && entry.type !== "avc3") {
    throw new Error(`unsupported codec box: ${entry.type}`);
  }
  // visual sample entry: 78 bytes of fixed fields before the child boxes
  const inner = new Box(stsd.view, entry.start + 8 + 78, entry.size - 8 - 78, "visual");
  inner.start -= 8; // children() expects start to point at a header; fake it
  inner.size += 8;
  const avcC = inner.find("avcC");
  if (!avcC) throw new Error("no avcC box");
  const desc = new Uint8Array(buf, avcC.bodyStart, avcC.size - 8);
  const codec = `avc1.${[desc[1], desc[2], desc[3]].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
  return { codec, description: desc };
}

// Parse an mp4 ArrayBuffer -> { codec, description, samples }, where samples
// is [{offset, size, pts (seconds), key}] in decode order.
export function parseMp4(buf) {
  const view = new DataView(buf);
  const root = new Box(view, -8, buf.byteLength + 8, "root"); // pseudo-box spanning the file
  const moov = root.find("moov");
  if (!moov) throw new Error("no moov box (fragmented or truncated mp4?)");
  const trak = videoTrak(moov);
  if (!trak) throw new Error("no video track");
  const mdhd = trak.path("mdia", "mdhd");
  const mdhdV = mdhd.view.getUint8(mdhd.bodyStart);
  const timescale = mdhd.view.getUint32(mdhd.bodyStart + (mdhdV === 1 ? 20 : 12));
  const stbl = trak.path("mdia", "minf", "stbl");
  const { codec, description } = avcConfig(stbl.find("stsd"), buf);

  // stsz: per-sample sizes
  const stsz = stbl.find("stsz");
  const uniform = stsz.view.getUint32(stsz.bodyStart + 4);
  const count = stsz.view.getUint32(stsz.bodyStart + 8);
  const sizes = uniform ? new Array(count).fill(uniform) : u32s(stsz, 12, count);

  // stts: decode timestamps (run-length {count, delta})
  const stts = stbl.find("stts");
  const dts = new Array(count);
  {
    const n = stts.view.getUint32(stts.bodyStart + 4);
    let t = 0, s = 0;
    for (let i = 0; i < n; i++) {
      const c = stts.view.getUint32(stts.bodyStart + 8 + i * 8);
      const d = stts.view.getUint32(stts.bodyStart + 12 + i * 8);
      for (let j = 0; j < c && s < count; j++, s++) {
        dts[s] = t;
        t += d;
      }
    }
  }

  // ctts: composition (pts) offsets — absent in our B-frame-free encodes
  const pts = dts.slice();
  const ctts = stbl.find("ctts");
  if (ctts) {
    const version = ctts.view.getUint8(ctts.bodyStart);
    const n = ctts.view.getUint32(ctts.bodyStart + 4);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const c = ctts.view.getUint32(ctts.bodyStart + 8 + i * 8);
      const raw = ctts.view.getUint32(ctts.bodyStart + 12 + i * 8);
      const off = version === 1 ? (raw | 0) : raw; // v1 is signed
      for (let j = 0; j < c && s < count; j++, s++) pts[s] = dts[s] + off;
    }
  }

  // stsc + stco/co64: chunk layout -> absolute byte offset per sample
  const stsc = stbl.find("stsc");
  const stco = stbl.find("stco");
  const co64 = stbl.find("co64");
  const chunkCount = (stco ?? co64).view.getUint32((stco ?? co64).bodyStart + 4);
  const chunkOffset = (i) =>
    stco ? stco.view.getUint32(stco.bodyStart + 8 + i * 4)
         : Number(co64.view.getBigUint64(co64.bodyStart + 8 + i * 8));
  const offsets = new Array(count);
  {
    const n = stsc.view.getUint32(stsc.bodyStart + 4);
    const runs = [];
    for (let i = 0; i < n; i++) {
      runs.push({
        firstChunk: stsc.view.getUint32(stsc.bodyStart + 8 + i * 12) - 1,
        perChunk: stsc.view.getUint32(stsc.bodyStart + 12 + i * 12),
      });
    }
    let s = 0;
    for (let r = 0; r < runs.length && s < count; r++) {
      const lastChunk = r + 1 < runs.length ? runs[r + 1].firstChunk : chunkCount;
      for (let c = runs[r].firstChunk; c < lastChunk && s < count; c++) {
        let off = chunkOffset(c);
        for (let j = 0; j < runs[r].perChunk && s < count; j++, s++) {
          offsets[s] = off;
          off += sizes[s];
        }
      }
    }
  }

  // stss: sync (key) samples — absent means every sample is a keyframe
  const stss = stbl.find("stss");
  let keys = null;
  if (stss) {
    keys = new Set(u32s(stss, 8, stss.view.getUint32(stss.bodyStart + 4)).map((x) => x - 1));
  }

  const samples = new Array(count);
  for (let i = 0; i < count; i++) {
    samples[i] = {
      offset: offsets[i],
      size: sizes[i],
      pts: pts[i] / timescale,
      key: keys ? keys.has(i) : true,
    };
  }
  return { codec, description, samples };
}
