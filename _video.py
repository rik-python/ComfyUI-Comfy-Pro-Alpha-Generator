"""Video probing and frame reading with PyAV (ships with ComfyUI)."""
import functools
import os

import av
import numpy as np


@functools.lru_cache(maxsize=32)
def _probe(path, mtime):
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 24)
        n = s.frames
        if not n:
            n = sum(1 for _ in c.decode(video=0))
        return {"frames": int(n), "fps": fps, "width": s.codec_context.width, "height": s.codec_context.height}


def probe(path):
    """frames / fps / width / height (cached per file version)."""
    return dict(_probe(path, os.path.getmtime(path)))


def read_frames(path, start=0, count=0):
    """uint8 RGB [count,H,W,3] from frame `start` (count 0 = to the end)."""
    out = []
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i < start:
                continue
            out.append(fr.to_ndarray(format="rgb24"))
            if count and len(out) == count:
                break
    if not out:
        raise ValueError(f"no frames read from {path} (start {start})")
    return np.stack(out)


def read_frame(path, index):
    return read_frames(path, index, 1)[0]
