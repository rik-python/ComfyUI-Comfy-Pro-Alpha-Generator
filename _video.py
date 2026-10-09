"""Plate helpers: image-sequence reading (Load Plate Sequence node) and the SAM3 proxy mp4.

Sequences: a folder of frames, or a pattern  D:/plates/sh010/plate.####.exr  /  plate.%04d.exr  (absolute, or
relative to ComfyUI/input). EXR (scene-linear) is converted with the sRGB curve and clipped, 16-bit PNG/TIFF is
scaled to 8 bit: LTX alpha-gen and SAM3 both work on display-referred images.
"""
import functools
import os
import re
import subprocess

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # read lazily by OpenCV at its first EXR access

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import folder_paths  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp", ".webp", ".dpx")


def _frame_number(name):
    m = re.search(r"(\d+)(?=\D*$)", os.path.splitext(name)[0])
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=32)
def _seq_files(spec, dir_mtime):
    if os.path.isdir(spec):
        folder, regex = spec, None
    else:
        folder, pattern = os.path.split(spec)
        rx = re.escape(pattern)
        rx = re.sub(r"(\\#|#)+", lambda _: r"\d+", rx)  # plate.####.exr
        rx = re.sub(r"%0?\d*d", lambda _: r"\d+", rx)   # plate.%04d.exr
        regex = re.compile(f"^{rx}$", re.IGNORECASE)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"image sequence folder not found: {folder}")
    files = [(_frame_number(f), os.path.join(folder, f)) for f in os.listdir(folder)
             if f.lower().endswith(IMAGE_EXT) and (regex is None or regex.match(f))]
    if not files:
        raise FileNotFoundError(f"no images found for {spec}")
    files.sort(key=lambda x: (x[0] is None, x[0] or 0, x[1]))
    return tuple(files)


def seq_spec(path):
    path = path.strip().strip('"')
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.get_input_directory(), path)
    return os.path.normpath(path)


def seq_files(path):
    """Sorted ((frame_number, file), ...) for a folder or ####/%04d pattern."""
    spec = seq_spec(path)
    folder = spec if os.path.isdir(spec) else os.path.dirname(spec)
    return _seq_files(spec, os.path.getmtime(folder) if os.path.isdir(folder) else 0)


def read_image(path):
    """uint8 RGB from PNG/JPG/TIF/EXR (EXR: scene-linear -> sRGB, clipped)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[..., :3]
    if img.dtype in (np.float32, np.float16, np.float64):
        x = np.clip(img.astype(np.float32), 0, 1)
        img = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
        img = (img * 255 + 0.5).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def proxy_mp4(images, fps, key):
    """IMAGE batch -> cached high-quality mp4 for SAM3's streaming tracker (SAM3 only needs to find the subject;
    LTX and the matte math use the images themselves)."""
    out_dir = os.path.join(folder_paths.get_temp_directory(), "selective_matte_seq")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{key}_{fps:.3f}.mp4")
    if os.path.isfile(out):
        return out
    n, h, w = images.shape[:3]
    # yuv420p needs even sizes: pad by edge replicate (SAM3 masks are cropped back by TrackMasks' resize/crop)
    ph, pw = h % 2, w % 2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w + pw}x{h + ph}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "12",
                             "-pix_fmt", "yuv420p", out + ".part.mp4"], stdin=subprocess.PIPE)
    for i in range(n):
        f = (images[i].float().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        if ph or pw:
            f = np.pad(f, ((0, ph), (0, pw), (0, 0)), mode="edge")
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed building the SAM3 proxy video")
    os.replace(out + ".part.mp4", out)
    return out
