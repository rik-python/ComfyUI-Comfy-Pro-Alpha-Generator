"""Selective-matte math (v12), ported from the CLI pipeline in workflows/matte/*.py.

Pure numpy / OpenCV / torch, no ComfyUI node code. Masks are float32 arrays [N,H,W] in 0-1, plates are uint8
or float arrays [N,H,W,3]. Function names and defaults match the CLI scripts so results stay comparable.
"""
import cv2
import numpy as np


def ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def to_gray(plate):
    """[N,H,W,3] float 0-1 or uint8 -> [N,H,W] uint8 (what DIS optical flow wants)."""
    if plate.dtype != np.uint8:
        plate = (np.clip(plate, 0, 1) * 255).astype(np.uint8)
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in plate])


# ---------------------------------------------------------------- selection (matte_pipeline.py prep)

def build_selection(target, adds=(), cuts=(), others=(), sub_grow=5):
    """target/adds/cuts/others: [N,H,W]. Returns (selection, others_mask).

    selection = (target | adds) minus grown, feathered cuts. Cuts (someone else's hand on the target) also count
    as "other people" so the isolated plate paints them out.
    """
    sel = target.astype(np.float32).copy()
    for a in adds:
        sel = np.maximum(sel, a)
    for c in cuts:
        for f in range(len(sel)):
            sel[f] *= 1.0 - cv2.GaussianBlur(cv2.dilate(c[f], ellipse(sub_grow)), (0, 0), 1.5)
    stack = list(others) + list(cuts)
    oth = np.maximum.reduce(stack) if stack else np.zeros_like(sel)
    return sel, oth.astype(np.float32)


# ---------------------------------------------------------------- v7 combine (combine_matte.py)

def combine(ltx, sam3, soft=1.5, other_thresh=0.9, floor=0.02, max_reach=150):
    """Nearest-owner split + connectivity reach (one frame)."""
    target = sam3 > 0.5
    other = (ltx > other_thresh) & ~target
    d_target = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    d_other = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 5)
    own = 1.0 / (1.0 + np.exp(np.clip((d_target - d_other) / soft, -50, 50)))
    alpha = ltx * own
    _, labels = cv2.connectedComponents(((alpha > floor) | target).astype(np.uint8), connectivity=8)
    attached = np.isin(labels, np.unique(labels[target])) & (d_target < max_reach)
    return alpha * attached


def separation_weight(ltx, sam3, reach=12, other_thresh=0.9, feather=3.0):
    """1 where the target's edge meets OTHER foreground (LTX solid on both sides, so LTX has no edge there)."""
    target = sam3 > 0.5
    other = (ltx > other_thresh) & ~target
    d_out = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)
    d_other = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 5)
    zone = ((d_out < reach) | (d_in < reach)) & (d_other < reach)
    return np.clip(cv2.GaussianBlur(zone.astype(np.float32), (0, 0), feather), 0.0, 1.0)


def hybrid(ltx, sam3, refined):
    """v7 frame: LTX edge against background, ViTMatte edge against other foreground."""
    w = separation_weight(ltx, sam3)
    return w * refined + (1.0 - w) * combine(ltx, sam3)


# ---------------------------------------------------------------- ViTMatte (onyx_style_vitmatte.py)

def trimap(sel, erode=10, dilate=15):
    sel = (sel > 0.5).astype(np.uint8)
    fg = cv2.erode(sel, ellipse(erode)).astype(bool)
    band = cv2.dilate(sel, ellipse(dilate)).astype(bool)
    return np.where(fg, 255, np.where(band, 128, 0)).astype(np.uint8)


def vitmatte_frame(model, proc, rgb, sel, erode=10, dilate=15, margin=64):
    """rgb uint8 [H,W,3], sel [H,W] -> alpha [H,W]. Crops to the subject like ONYX does."""
    import torch
    from PIL import Image

    tri = trimap(sel, erode, dilate)
    alpha = np.zeros(tri.shape, np.float32)
    ys, xs = np.where(tri > 0)
    if len(ys) == 0:
        return alpha
    h, w = tri.shape
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin)
    inp = proc(images=Image.fromarray(rgb[y0:y1, x0:x1]), trimaps=Image.fromarray(tri[y0:y1, x0:x1]),
               return_tensors="pt").to(model.device, model.dtype)
    with torch.no_grad():
        pred = model(**inp).alphas[0, 0, : y1 - y0, : x1 - x0].float().cpu().numpy()
    alpha[y0:y1, x0:x1] = pred
    alpha[tri == 255], alpha[tri == 0] = 1.0, 0.0
    return alpha


# ---------------------------------------------------------------- temporal stabilize (temporal_stabilize.py)

def stabilize(seqs, gray, radius=2, t_sigma=1.2, photo_sigma=8.0, progress=None):
    """seqs: list of [N,H,W] float arrays sharing the plate's motion. Returns stabilized copies."""
    n, h, w = gray.shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = [np.empty_like(s) for s in seqs]
    for t in range(n):
        acc = [s[t].copy() for s in seqs]
        wsum = np.ones((h, w), np.float32)
        for k in range(max(0, t - radius), min(n, t + radius + 1)):
            if k == t:
                continue
            flow = dis.calc(gray[t], gray[k], None)
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            warped_plate = cv2.remap(gray[k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err = cv2.GaussianBlur(np.abs(gray[t].astype(np.float32) - warped_plate.astype(np.float32)), (0, 0), 2)
            wk = np.exp(-((k - t) ** 2) / (2 * t_sigma ** 2)) * np.exp(-(err / photo_sigma) ** 2)
            for i, s in enumerate(seqs):
                acc[i] += wk * cv2.remap(s[k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wsum += wk
        for i in range(len(seqs)):
            out[i][t] = acc[i] / wsum
        if progress:
            progress(t + 1, n)
    return out


# ---------------------------------------------------------------- isolated plate (isolate_plate.py, inpaint fill)

def isolate_frame(rgb, tgt, others, grow=6, blur=6.0):
    """rgb float 0-255 [H,W,3]. Paint every OTHER person with inpainted background so LTX sees only the target."""
    h, w = tgt.shape
    if grow:
        others = cv2.dilate(others, ellipse(grow))
    paint = np.clip(others * (1.0 - tgt), 0, 1)
    if paint.max() <= 0.02:
        return rgb
    hole = ((paint > 0.02) | (tgt > 0.02)).astype(np.uint8)
    small = cv2.resize(rgb.astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small_hole = cv2.dilate(cv2.resize(hole, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, small_hole, 8, cv2.INPAINT_TELEA)
    src = cv2.GaussianBlur(cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32), (0, 0), blur)
    return rgb * (1 - paint[..., None]) + src * paint[..., None]


# ---------------------------------------------------------------- v11 level (level_isolated.py)

def level(ltx, target_soft, others_soft, ramp=6, floor=0.05):
    tgt = target_soft > 0.5
    paint_soft = np.clip(others_soft * (1.0 - target_soft), 0, 1)
    paint = paint_soft > 0.5
    core = cv2.erode(paint.astype(np.uint8), ellipse(15)).astype(bool)
    g = float(np.median(ltx[core])) if core.any() else 0.0
    near = cv2.GaussianBlur(cv2.dilate(paint.astype(np.uint8), ellipse(15)).astype(np.float32), (0, 0), 4)
    out = near * np.clip((ltx - g) / (1 - g), 0, 1) + (1 - near) * ltx
    d_tgt = cv2.distanceTransform((~tgt).astype(np.uint8), cv2.DIST_L2, 5)
    out *= 1.0 - paint_soft * np.clip((d_tgt - ramp) / 2.0, 0, 1)
    _, labels = cv2.connectedComponents(((out > floor) | tgt).astype(np.uint8), connectivity=8)
    return out * np.isin(labels, np.unique(labels[tgt])), g


# ---------------------------------------------------------------- v12 blend (matte_pipeline.py finish)

def v12_blend(v7, v11, sel, others, near_px=15):
    paint = ((others > .5) & ~(sel > .5)).astype(np.uint8)
    wgt = cv2.GaussianBlur(cv2.dilate(paint, ellipse(near_px)).astype(np.float32), (0, 0), 5)
    return wgt * v7 + (1 - wgt) * v11


def ltx_fallback(alpha, ltx, sel, refined, core_px=10, reach=25.0, lo=0.3, hi=0.7, down=8):
    """Per AREA: how much LTX covers the selection's core around each pixel (blur radius ~reach px). Where LTX
    misses part of the selection (it only mattes what it reads as the main subject: a table, a parked car), blend
    that part toward the ViTMatte matte; parts LTX covers (the person holding the glass) keep LTX edges.
    People: cover ~1, alpha unchanged. Works at 1/down resolution (the map is smooth anyway).
    Returns (alpha, cover per frame)."""
    n, h, w = alpha.shape
    sh, sw = max(1, h // down), max(1, w // down)
    num = np.zeros((n, sh, sw), np.float32)
    den = np.zeros((n, sh, sw), np.float32)
    cover = np.ones(n, np.float32)
    for f in range(n):
        s = sel[f] > 0.5
        core = cv2.erode(s.astype(np.uint8), ellipse(core_px)).astype(bool)
        core = core if core.any() else s
        if not core.any():
            continue
        cover[f] = float(ltx[f][core].mean())
        c = cv2.resize(core.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        l = cv2.resize(ltx[f] * core, (sw, sh), interpolation=cv2.INTER_AREA)
        num[f] = cv2.GaussianBlur(l, (0, 0), reach / down)
        den[f] = cv2.GaussianBlur(c, (0, 0), reach / down)
    if n > 2:  # smooth over time so the fallback never pops between frames
        pad = lambda a: np.concatenate([a[:1], a, a[-1:]])
        num, den = [(lambda p: (p[:-2] + 2 * p[1:-1] + p[2:]) / 4)(pad(a)) for a in (num, den)]
    out = np.empty_like(alpha)
    for f in range(n):
        local = np.where(den[f] > 1e-3, num[f] / np.maximum(den[f], 1e-3), 1.0)
        wgt = cv2.resize(np.clip((local - lo) / (hi - lo), 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
        out[f] = wgt * alpha[f] + (1 - wgt) * refined[f]
    return out, cover


# ---------------------------------------------------------------- LTX chunking (run_ltx_chunks.py)

def chunk_starts(n, chunk, stride):
    if n <= chunk:
        return [0]
    starts = list(range(0, n - chunk + 1, stride))
    if starts[-1] != n - chunk:
        starts.append(n - chunk)
    return starts


def chunk_weight(i, starts, chunk, k):
    """Triangular cross-fade weight of frame k (0-based inside chunk i)."""
    s = starts[i]
    ramp_in = 1.0 if i == 0 else min(1.0, (k + 1) / (starts[i - 1] + chunk - s + 1))
    ramp_out = 1.0 if i == len(starts) - 1 else min(1.0, (chunk - k) / (s + chunk - starts[i + 1] + 1))
    return min(ramp_in, ramp_out)


def ltx_frames(n):
    """Largest valid LTX length (8k+1) <= n."""
    return max(1, (n - 1) // 8 * 8 + 1)


# ---------------------------------------------------------------- QC (measure_matte.py)

def qc(gray, sel, ltx, alpha, others=None):
    """Motion-compensated flicker ratio vs LTX alone + share of LTX hair kept. Returns dict."""
    n, h, w = gray.shape
    if n < 3:
        return {}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fl_ltx, fl_out = [], []
    for t in range(1, n - 1):
        T = (sel[t] > .5).astype(np.uint8)
        band = cv2.dilate(T, np.ones((41, 41), np.uint8)).astype(bool) & ~cv2.erode(T, np.ones((41, 41), np.uint8)).astype(bool)
        maps, good = [], np.ones((h, w), bool)
        for k in (t - 1, t + 1):
            fl = dis.calc(gray[t], gray[k], None)
            mx, my = gx + fl[..., 0], gy + fl[..., 1]
            maps.append((mx, my))
            err = np.abs(gray[t].astype(np.float32) - cv2.remap(gray[k], mx, my, cv2.INTER_LINEAR).astype(np.float32))
            good &= cv2.GaussianBlur(err, (0, 0), 2) < 8
        m = band & good
        if not m.any():
            continue
        for q, acc in ((ltx, fl_ltx), (alpha, fl_out)):
            pred = 0.5 * (cv2.remap(q[t - 1], *maps[0], cv2.INTER_LINEAR) + cv2.remap(q[t + 1], *maps[1], cv2.INTER_LINEAR))
            acc.append(np.abs(q[t] - pred)[m].mean())
    tot = kept = 0.0
    for f in range(n):
        T = sel[f] > .5
        O = (ltx[f] > .9) & ~T
        dT = cv2.distanceTransform((~T).astype(np.uint8), cv2.DIST_L2, 5)
        dO = cv2.distanceTransform((~O).astype(np.uint8), cv2.DIST_L2, 5)
        hs = (ltx[f] > .02) & (ltx[f] < .9) & (dT < dO)
        tot += ltx[f][hs].sum()
        kept += alpha[f][hs].sum()
    base = float(np.mean(fl_ltx)) if fl_ltx else 0.0
    return {"flicker_x": float(np.mean(fl_out)) / base if base else 0.0,
            "hair_pct": float(kept / tot * 100) if tot else 100.0}
