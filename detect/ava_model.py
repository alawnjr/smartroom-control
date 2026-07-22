#!/usr/bin/env python3
"""SlowFast-AVA spatiotemporal action detection, shared by the live service
(live_infer.py) and the batch action pass (action.py --variant ava).

Unlike the skeleton recognizers (ST-GCN++/PoseC3D), this model reads RGB
pixels, not keypoints: it takes a short video clip plus the person boxes in its
final frame as proposals, and returns per-box multi-label scores over the 60
AVA atomic actions. One forward classifies everyone in the frame. Poses are
still computed and drawn by the caller — they just no longer drive the label.

Config (env): SMARTROOM_AVA_CONFIG, SMARTROOM_AVA_CKPT, SMARTROOM_AVA_LABELS,
SMARTROOM_AVA_THR, SMARTROOM_AVA_BLACKLIST.
"""

import contextlib
import os
import threading
from pathlib import Path

AVA_SHORT = 256   # short side the clip is resized to before the forward pass
# Multi-label sigmoid scores: every class above this is reported, not just top-1.
AVA_THR = float(os.environ.get("SMARTROOM_AVA_THR", "0.4"))
# Classes to suppress entirely (never output). ';'-separated (AVA names contain
# commas), case-insensitive exact match. Override/extend via SMARTROOM_AVA_BLACKLIST.
AVA_BLACKLIST = {s.strip().lower() for s in
                 os.environ.get("SMARTROOM_AVA_BLACKLIST",
                                "watch (a person);talk to (e.g., self, a person, a group)").split(";")
                 if s.strip()}

# mmaction's registry is process-global and not thread-safe to populate: two
# models building at once raced into "MaxIoUAssignerAVA is not in the registry".
_BUILD_LOCK = threading.Lock()


def load_label_map(path):
    """AVA label map: 'id: name' per line -> {int id: name} (same as the demo)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ": " in line:
            i, name = line.split(": ", 1)
            out[int(i)] = name.strip()
    return out


def resize_short(w, h, short=AVA_SHORT):
    """New (w, h) with the short side scaled to `short`, aspect preserved."""
    scale = short / min(w, h)
    return int(round(w * scale)), int(round(h * scale))


def default_paths():
    """(config, checkpoint, label_map) for the pretrained SlowFast-AVA 2.1."""
    import mmaction
    cfg = os.environ.get("SMARTROOM_AVA_CONFIG") or os.path.join(
        os.path.dirname(mmaction.__file__), ".mim", "configs", "detection",
        "slowfast", "slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py")
    ckpt = os.environ.get("SMARTROOM_AVA_CKPT") or str(
        Path.home() / "Code/yolo-bench/slowfast_ava.pth")
    labels = os.environ.get("SMARTROOM_AVA_LABELS") or str(
        Path(__file__).resolve().parent / "ava_label_map.txt")
    return cfg, ckpt, labels


@contextlib.contextmanager
def _on_device(device):
    """Make `device` the calling thread's CURRENT cuda device for the block.

    Putting tensors on a device is not the same as making it current. cuDNN and
    cuBLAS handles are cached per device but acquired against whatever device
    the *thread* currently has selected, which defaults to cuda:0 for every new
    thread. Running a model that lives on cuda:2 from a thread still pointed at
    cuda:0 therefore hands cuDNN a handle from the wrong device — reported as
    CUDNN_STATUS_MAPPING_ERROR — and corrupts cuda:0's context in the process.
    That is not a local failure: every later kernel on cuda:0 dies with "CUDA
    error: misaligned address", including the pose model of an unrelated
    camera, and a poisoned context never recovers. It is what took the live
    service down.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        yield
        return
    if not isinstance(device, str) or not device.startswith("cuda"):
        yield
        return
    with torch.cuda.device(torch.device(device)):
        yield


class AvaDetector:
    """A loaded SlowFast-AVA model. Build once, call `infer` per clip window."""

    def __init__(self, config_path=None, ckpt=None, label_map_path=None,
                 device="cpu", thr=AVA_THR):
        import mmengine
        import numpy as np
        from mmengine.runner import load_checkpoint
        from mmaction.registry import MODELS
        try:
            from mmaction.utils import register_all_modules
            register_all_modules(True)
        except Exception:  # noqa: BLE001
            pass

        d_cfg, d_ckpt, d_lm = default_paths()
        config_path = config_path or d_cfg
        ckpt = ckpt or d_ckpt
        label_map_path = label_map_path or d_lm

        # Everything below must run with THIS device current, not just with the
        # tensors on it — see the note on `infer`.
        with _BUILD_LOCK, _on_device(device):
            cfg = mmengine.Config.fromfile(config_path)
            # equal bbox count across classes (as the demo does); test_cfg.rcnn
            # may be None in the config, so build the dict rather than index it.
            tc = cfg.model.get("test_cfg") or {}
            tc["rcnn"] = dict(action_thr=0)
            cfg.model["test_cfg"] = tc
            cfg.model.backbone.pretrained = None
            model = MODELS.build(cfg.model)
            load_checkpoint(model, ckpt, map_location="cpu")
            model.to(device).eval()

        sampler = [x for x in cfg.val_pipeline
                   if str(x["type"]).endswith("SampleAVAFrames")][0]
        self.model = model
        self.device = device
        self.thr = thr
        self.clip_len = sampler["clip_len"]
        self.interval = sampler["frame_interval"]
        self.mean = np.array(cfg.model.data_preprocessor["mean"])
        self.std = np.array(cfg.model.data_preprocessor["std"])
        self.label_map = load_label_map(label_map_path)

    def infer(self, frames_small, boxes_scaled):
        """One forward pass.

        frames_small: list of BGR frames already resized to the AVA short side.
                      Resampled to clip_len, so any length/rate works.
        boxes_scaled: [(key, [x1,y1,x2,y2])] proposals in those resized coords.

        Returns {key: [[class_name, score], ...]} sorted by score, above
        threshold and with blacklisted classes removed (so an empty list means
        "nothing confident", not an error).
        """
        import mmcv
        import numpy as np
        import torch
        from mmengine.structures import InstanceData
        from mmaction.structures import ActionDataSample

        if not frames_small or not boxes_scaled:
            return {}
        nh, nw = frames_small[-1].shape[:2]
        idx = np.linspace(0, len(frames_small) - 1, self.clip_len).round().astype(int)
        imgs = [frames_small[i].astype(np.float32) for i in idx]
        for im in imgs:
            mmcv.imnormalize_(im, self.mean, self.std, to_rgb=False)
        arr = np.stack(imgs).transpose(3, 0, 1, 2)[np.newaxis]   # 1,C,T,H,W
        inp = torch.from_numpy(arr).to(self.device)
        keys = [k for k, _ in boxes_scaled]
        prop = torch.tensor([b for _, b in boxes_scaled], dtype=torch.float32,
                            device=self.device)
        ds = ActionDataSample()
        ds.proposals = InstanceData(bboxes=prop)
        ds.set_metainfo(dict(img_shape=(nh, nw)))
        with torch.no_grad(), _on_device(self.device):
            res = self.model(inp, [ds], mode="predict")
        scores = res[0].pred_instances.scores    # (num_proposals, num_classes)

        out = {}
        for j, key in enumerate(keys):
            labs = [(self.label_map[i], float(scores[j, i]))
                    for i in range(scores.shape[1])
                    if i in self.label_map and float(scores[j, i]) > self.thr
                    and self.label_map[i].lower() not in AVA_BLACKLIST]
            labs.sort(key=lambda x: -x[1])
            out[key] = [[a, round(s, 3)] for a, s in labs]
        return out
