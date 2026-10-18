"""Comfy Pro Alpha Generator math (v12), ported from the CLI pipeline in workflows/matte/*.py.

Pure numpy / OpenCV / torch, no ComfyUI node code. Masks are float32 arrays [N,H,W] in 0-1, plates are uint8
or float arrays [N,H,W,3]. Function names and defaults match the CLI scripts so results stay comparable.
"""
import math

import cv2
import numpy as np


def _f32(a):
    """One frame as float32 (stage outputs are stored as uint8 / float16 to save RAM; OpenCV wants float32)."""
    return np.asarray(a, dtype=np.float32)


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


def attn_mask(m, grow=20, blur=10.0):
    """SAM mask -> attention mask for the IC-LoRA guide (v13): grown so hair outside the SAM outline still gets
    attention, feathered so it fades out. A hard SAM mask cuts the hair (tested: 1.5x the error)."""
    m = cv2.dilate(_f32(m), ellipse(grow)) if grow else _f32(m)
    return cv2.GaussianBlur(m, (0, 0), blur) if blur else m


def _local_mean(v, m, sigma, fallback):
    """Gaussian-weighted mean of v over mask m around every pixel (fallback where m has no support)."""
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    return np.where(w > 1e-3, cv2.GaussianBlur(v * m, (0, 0), sigma) / np.maximum(w, 1e-3), fallback)


def front_map(ct, co, vis_t, vis_o, grow=20, min_area=200, reach=15.0, depth=None, depth_reach=30.0):
    """v13: where the other subjects are IN FRONT of the target, as a soft weight.

    ct / co: target / others mattes from the attention passes, clipped by LTX on the plate. These are amodal: LTX
    also draws the parts each one hides. Where both are > .5 one hides the other, and whoever SAM sees there is in
    front (agreed with a Depth Anything pass on 97-100% of overlaps). The vote is LOCAL (within ~`reach` px), not
    per overlap blob: one blob can hold the target's real arm in front of someone AND an invented arm behind them."""
    both = (ct > .5) & (co > .5)
    b = both.astype(np.float32)
    vt, vo = vis_t > .5, vis_o > .5
    so = cv2.GaussianBlur(b * vo, (0, 0), reach)
    st = cv2.GaussianBlur(b * vt, (0, 0), reach)
    # SAM's own answer wins where it is clear; pixels SAM gives to both or neither are settled by depth when there
    # is a depth pass (a hand in front of a chest: SAM's coarse masks give the fingers to both people, a local vote
    # lets the big chest win), else by the local vote. (Making the target's selection final instead broke hair over
    # the target: SAM often includes it, GT error 4x.)
    unclear = vo == vt
    if depth is not None:
        d = _f32(depth)
        only_t, only_o = vt & ~vo, vo & ~vt
        dt = _local_mean(d, only_t, depth_reach, float(np.median(d[only_t])) if only_t.any() else 0.0)
        do = _local_mean(d, only_o, depth_reach, float(np.median(d[only_o])) if only_o.any() else 1.0)
        near_o = both & (np.abs(d - do) < np.abs(d - dt))
        # vote the per-pixel depth answers over ~25 px: depth blurs thin fast fingers into what is behind them, the
        # palm's clear answer carries to the tips (far-apart regions still decide separately)
        settle = cv2.GaussianBlur(near_o.astype(np.float32), (0, 0), 25) > cv2.GaussianBlur((both & ~near_o).astype(np.float32), (0, 0), 25)
    else:
        settle = so > st
    front = (both & ((vo & ~vt) | (unclear & settle))).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(front, connectivity=8)
    small = np.isin(lab, np.nonzero(stats[:, cv2.CC_STAT_AREA] < min_area)[0])
    front[small & (lab > 0)] = 0
    # grow into the others' soft edge (strands, fingers) but never into overlap the target won: there the others'
    # matte is the part they hide BEHIND the target, and subtracting it would bite holes in the target
    back = cv2.GaussianBlur((both & ~front.astype(bool)).astype(np.float32), (0, 0), 2)
    # HARD reach: full weight on everything within `grow` px of an in-front pixel, so a fast finger's tip (outside
    # the overlap, where the target pass is already soft) is subtracted fully, not half (a wide blur ghosted it)
    return cv2.GaussianBlur(cv2.dilate(front, ellipse(grow)).astype(np.float32), (0, 0), 2) * (1 - back)


def guided(I, p, r=4, eps=1e-3):
    """Grey guided filter (He et al.): `p` re-fitted to the edges of image `I` (both 0-1) within ~r px."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    return box(a) * I + box(mp - a * mI)


def v13_overlap(ltx, attn_t, attn_o, sel, others, out, down=4, near=50, far=90, depth=None, tsmooth=3, grow=20,
                gray=None, guide_r=4, guide_eps=1e-3):
    """v13 matte: target attention pass minus the others' attention pass wherever the others are in front.

    Both passes run on the REAL plate (no paint-out), so hair and fingers crossing the target keep their own
    generative edge. The front/back weights are smoothed over 5 frames at 1/`down` resolution (they are blurry
    anyway), so no decision pops between frames and RAM stays low. The target matte fades out between `near` and
    `far` px from its SAM selection: anything further is a limb LTX invented, even where nobody covers it.
    `gray` (optional, [N,H,W] uint8 or 0-1): plate luma; when given, the others' matte is re-fitted to the plate's
    edges where it is subtracted (sharper strands over the target, the node's "sharpen hair" switch)."""
    n, h, w = out.shape
    sh, sw = max(1, h // down), max(1, w // down)
    wts = np.empty((n, sh, sw), np.float32)
    for f in range(n):
        lt = _f32(ltx[f])
        ct, co = np.minimum(_f32(attn_t[f]), lt), np.minimum(_f32(attn_o[f]), lt)
        dp = None if depth is None else depth[f]
        wts[f] = cv2.resize(front_map(ct, co, _f32(sel[f]), _f32(others[f]), grow=grow, depth=dp), (sw, sh),
                            interpolation=cv2.INTER_AREA)
    if tsmooth > 1:
        k = tsmooth // 2
        pad = np.concatenate([wts[:1]] * k + [wts] + [wts[-1:]] * k)
        wts = sum(pad[i:i + n] for i in range(tsmooth)) / tsmooth
    for f in range(n):
        lt = _f32(ltx[f])
        ct, co = np.minimum(_f32(attn_t[f]), lt), np.minimum(_f32(attn_o[f]), lt)
        wt = np.clip(cv2.resize(wts[f], (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
        if gray is not None and wt.max() > .05:
            # LTX draws an occluder's hair over the target soft (~.5 between curls, strands smeared): re-fit the
            # others' matte to the plate's own edges (dark strand / bright shirt) where it is subtracted
            g = _f32(gray[f]) / (255.0 if gray.dtype == np.uint8 else 1.0)  # node passes uint8 (RAM), harness 0-1
            co = np.where(wt > .05, np.clip(guided(g, co, guide_r, guide_eps), 0, 1), co)
        d = cv2.distanceTransform((_f32(sel[f]) <= .5).astype(np.uint8), cv2.DIST_L2, 5)
        out[f] = np.clip(ct * np.clip((far - d) / (far - near), 0, 1) - wt * co, 0, 1)
    return out


def ltx_fallback(alpha, ltx, sel, refined, core_px=10, reach=25.0, lo=0.3, hi=0.7, down=8, out=None):
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
        lf = _f32(ltx[f])
        cover[f] = float(lf[core].mean())
        c = cv2.resize(core.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        l = cv2.resize(lf * core, (sw, sh), interpolation=cv2.INTER_AREA)
        num[f] = cv2.GaussianBlur(l, (0, 0), reach / down)
        den[f] = cv2.GaussianBlur(c, (0, 0), reach / down)
    if n > 2:  # smooth over time so the fallback never pops between frames
        pad = lambda a: np.concatenate([a[:1], a, a[-1:]])
        num, den = [(lambda p: (p[:-2] + 2 * p[1:-1] + p[2:]) / 4)(pad(a)) for a in (num, den)]
    out = np.empty_like(alpha) if out is None else out
    for f in range(n):
        local = np.where(den[f] > 1e-3, num[f] / np.maximum(den[f], 1e-3), 1.0)
        wgt = cv2.resize(np.clip((local - lo) / (hi - lo), 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out[f] = wgt * _f32(alpha[f]) + (1 - wgt) * _f32(refined[f])
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


# ---------------------------------------------------------------- v14 native-resolution edges (v14_tile_spike.py)
# A plate bigger than LTX's native 1920x1088 is matted on a proxy that fits inside it. Native-resolution LTX tiles
# then supply the edges: the proxy result keeps the subject's core and every decision, the tiles add strands.

TILE_W, TILE_H = 1920, 1088


def _box(w, h):
    """LTX's native frame in the plate's orientation: 1920x1088, or 1088x1920 for portrait plates (v0.3.0 ran a
    vertical 1080p plate directly at 1088x1920)."""
    return (TILE_W, TILE_H) if w >= h else (TILE_H, TILE_W)


def needs_proxy(w, h):
    bw, bh = _box(w, h)
    return w > bw or h > bh


def proxy_size(w, h, max_w=None, max_h=None):
    """Largest even size with the plate's aspect that fits inside max_w x max_h (default: LTX's frame in the
    plate's orientation), never larger than the plate."""
    if max_w is None:
        max_w, max_h = _box(w, h)
    scale = min(max_w / w, max_h / h, 1.0)
    return max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)


def grid_axis(n, t):
    """Tile start positions along one axis: >= 50% overlap, first at 0, last flush with the plate edge."""
    if n <= t:
        return [0]
    k = math.ceil((n - t) / (t / 2)) + 1
    return [round(i * (n - t) / (k - 1)) for i in range(k)]


def tile_grid(w, h, tw=None, th=None):
    """(x, y, tile_w, tile_h) for every tile, rows top to bottom; tiles follow the plate's orientation; a plate
    axis shorter than a tile gets one tile of the plate's size there (LTX pads it)."""
    if tw is None:
        tw, th = _box(w, h)
    xs, ys = grid_axis(w, tw), grid_axis(h, th)
    return [(x, y, min(tw, w), min(th, h)) for y in ys for x in xs]


def select_tiles(sel, grid, native_wh, band=30):
    """Indices of the tiles that touch the subject's edge band (+-band proxy px around the selection's outline)
    on any frame. sel: proxy selection [n, ph, pw]."""
    n, ph, pw = sel.shape
    W, H = native_wh
    k = ellipse(band)
    edge = np.zeros((ph, pw), bool)
    for f in range(n):
        s = (_f32(sel[f]) > .5).astype(np.uint8)
        if s.any():
            edge |= cv2.dilate(s, k).astype(bool) & ~cv2.erode(s, k).astype(bool)
    sx, sy = pw / W, ph / H
    return [i for i, (x, y, tw, th) in enumerate(grid)
            if edge[int(y * sy):math.ceil((y + th) * sy), int(x * sx):math.ceil((x + tw) * sx)].any()]


def _ramp(n, lo, hi):
    """1-D blend weight: cosine rise over `lo` px at the start, fall over `hi` px at the end, 1 elsewhere."""
    r = np.ones(n, np.float32)
    for ov, sl in ((lo, slice(0, lo)), (hi, slice(n - hi, n))):
        if ov > 0:
            t = (0.5 - 0.5 * np.cos(np.pi * (np.arange(ov) + 0.5) / ov)).astype(np.float32)
            r[sl] = np.minimum(r[sl], t if sl.start == 0 else t[::-1])
    return r


def tile_feather(grid, idx, native_wh):
    """Stitch weight of tile `idx`: ramps only where it overlaps a neighbour (over half the overlap), flat at
    the plate border."""
    x, y, tw, th = grid[idx]
    xs, ys = sorted({g[0] for g in grid}), sorted({g[1] for g in grid})
    i, j = xs.index(x), ys.index(y)
    left = (xs[i - 1] + tw - x) // 2 if i > 0 else 0
    right = (x + tw - xs[i + 1]) // 2 if i < len(xs) - 1 else 0
    top = (ys[j - 1] + th - y) // 2 if j > 0 else 0
    bottom = (y + th - ys[j + 1]) // 2 if j < len(ys) - 1 else 0
    return np.outer(_ramp(th, top, bottom), _ramp(tw, left, right))


def stitch_tiles(frames, grid, used, native_wh, feathers):
    """One frame: feather-blend the rendered tiles (`frames[i]` is tile `used[i]`, tile-sized) into the native
    frame. coverage = rendered weight / whole-grid weight, so it fades out where a rendered tile overlaps one that
    was not rendered (no step at that tile border). feathers: tile_feather for every grid tile."""
    W, H = native_wh
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for a, i in zip(frames, used):
        x, y, tw, th = grid[i]
        acc[y:y + th, x:x + tw] += _f32(a)[:th, :tw] * feathers[i]
        wsum[y:y + th, x:x + tw] += feathers[i]
    if not used:
        return acc, wsum
    full = np.zeros((H, W), np.float32)
    for (x, y, tw, th), f in zip(grid, feathers):
        full[y:y + th, x:x + tw] += f
    return acc / np.maximum(wsum, 1e-6), np.clip(wsum / full, 0, 1)


def merge_native(base, tiles, coverage, s, others=None):
    """One frame of "edges from tiles". base: proxy alpha scaled up to native (0..1); tiles: stitched tile alpha;
    coverage: 0..1, 0 where no tile; s = native / proxy width; others: the other subjects' mask at native size.
    The proxy keeps the solid core, everything near other people (a tile only sees part of the frame and decides
    worse there: v13 crossings, leaks onto a neighbour) and wherever the tiles disagree over an area; elsewhere the
    tiles' native strands are used. Constants tuned at 4K (s = 2)."""
    k = s / 2
    b, t = _f32(base), _f32(tiles)
    core = cv2.erode((b > .98).astype(np.uint8), ellipse(max(1, round(6 * k)))).astype(np.float32)
    core = cv2.GaussianBlur(core, (0, 0), 3 * k)
    trust = np.clip((0.35 - cv2.GaussianBlur(np.abs(t - b), (0, 0), 12 * k)) / 0.2, 0, 1) * _f32(coverage)
    # tiles only refine edges near the subject: a faint haze a tile leaves on someone else, far from the proxy's
    # matte, "agrees" with base 0 closely enough to pass the trust test otherwise (seen on plate b's denim jacket)
    near = cv2.dilate((b > .02).astype(np.uint8), ellipse(max(1, round(20 * k)))).astype(np.float32)
    trust *= cv2.GaussianBlur(near, (0, 0), 4 * k)
    if others is not None:
        near_o = cv2.dilate((_f32(others) > .5).astype(np.uint8), ellipse(max(1, round(24 * k)))).astype(np.float32)
        trust *= 1 - cv2.GaussianBlur(near_o, (0, 0), 4 * k)
    return np.clip(core * b + (1 - core) * (trust * t + (1 - trust) * b), 0, 1)


# ---------------------------------------------------------------- v15 edge colour

def srgb_to_linear(x):
    """sRGB-encoded 0-1 -> linear light (IEC 61966-2-1), float32."""
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    """Linear light 0-1 -> sRGB-encoded, float32."""
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def blur_fusion(I, a, r, F=None, B=None):
    """One Blur-Fusion pass (Forte, ICIP 2021). I: [H,W,3] linear plate; a: [H,W]; F, B: the previous pass's
    estimates (default: the plate). Returns (F, B): the foreground estimate and the blurred background."""
    F = I if F is None else F
    B = I if B is None else B
    a3 = _f32(a)[..., None]
    ba = cv2.blur(_f32(a), (r, r))[..., None]
    fb = cv2.blur(F * a3, (r, r)) / (ba + 1e-5)
    bb = cv2.blur(B * (1 - a3), (r, r)) / (1 - ba + 1e-5)
    return np.clip(fb + a3 * (I - a3 * fb - (1 - a3) * bb), 0, 1).astype(np.float32), bb.astype(np.float32)


LUMA = np.float32([0.2126, 0.7152, 0.0722])  # Rec.709 luminance weights (linear light)


def _blur_big(x, sigma):
    """Gaussian blur that stays fast at large sigma: work on a downscaled copy (sigma 2 there), scale back up."""
    if sigma <= 4:
        return cv2.GaussianBlur(x, (0, 0), sigma).reshape(x.shape)  # OpenCV drops a single channel axis
    h, w = x.shape[:2]
    f = sigma / 2
    sw, sh = max(1, round(w / f)), max(1, round(h / f))
    small = cv2.GaussianBlur(cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).reshape(x.shape)


def fill_from(x, mask, s=1.0, default=None):
    """Values of x inside mask pushed outward (normalized convolution, finest scale that reaches each pixel):
    the EdgeExtend colour of the solid inside. x: [H,W,C]; returns float32 [H,W,C]. Where nothing reaches:
    `default` (per channel), or x itself."""
    out = np.array(x, np.float32, copy=True)
    if default is not None:
        out[:] = np.float32(default)
    done = np.zeros(mask.shape, bool)
    m = mask.astype(np.float32)
    xm = x * m[..., None]
    for sg in (1.5, 3, 6, 12, 24, 48, 96):
        sg *= max(s, 0.25)
        den = _blur_big(m, sg)
        ok = (den > 0.02) & ~done
        if ok.any():
            out[ok] = _blur_big(xm, sg)[ok] / den[ok][:, None]
            done |= ok
        if done.all():
            break
    return out


def quick_colour(I, a, s=1.0, erode=2, spread=1.5, near=24, far=40):
    """v15 quick: the subject's own colour F in the soft edge, one frame, linear light.

    The edge band is everything outside the solid inside (alpha >= 0.98, eroded by `erode` px at 1080p): LTX
    edges often run a pixel wide, so even alpha ~1 there can still carry sky or grass. In the band the colour
    (chroma) comes from the solid inside, pushed outward (EdgeExtend), and the brightness comes from Blur-Fusion
    (Forte, ICIP 2021; r = 90 then 6 at 1080p; s = frame width / 1920) so strand and fabric texture stays,
    clamped to the inside's local brightness range (mean +- spread x std) so a white sky or a dark background
    cannot leak in as brightness. The solid inside is the plate exactly.

    Only pixels near the real background (alpha < 0.05) are rebuilt: fully within `near` px (at 1080p), fading to
    the plate by `far` px. LTX leaves fuzzy props (a bouquet) semi-solid deep inside the subject, where no
    background shows through; rebuilding those smears them. Measured: hair edges lie within 20 to 32 px of the
    background, bouquet interiors 50 px and more."""
    I = _f32(I)
    a = _f32(a)
    F, B = blur_fusion(I, a, max(1, round(90 * s)))
    F, _ = blur_fusion(I, a, max(1, round(6 * s)), F, B)
    inner = cv2.erode((a >= 0.98).astype(np.uint8), ellipse(max(1, round(erode * s)))).astype(bool)
    if not inner.any():  # nothing solid to extend from: Blur-Fusion alone
        return F
    Fe = fill_from(I, inner, s)
    L = I @ LUMA
    st = fill_from(np.dstack([L, L * L]), inner, s)
    sd = np.sqrt(np.maximum(st[..., 1] - st[..., 0] ** 2, 0))
    Lt = np.clip(F @ LUMA, st[..., 0] - spread * sd, st[..., 0] + spread * sd)
    out = np.clip(Fe * (Lt / np.maximum(Fe @ LUMA, 1e-4))[..., None], 0, 1)
    d = cv2.distanceTransform((a >= 0.05).astype(np.uint8), cv2.DIST_L2, 5) / max(s, 1e-3)
    wt = (1 - _smoothstep(near, far, d))[..., None]
    out = wt * out + (1 - wt) * I
    return np.where(inner[..., None] | (wt == 0), I, out).astype(np.float32)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _ring(a, s):
    """The visible background just outside the matte: alpha < 0.02 within 12 px (at 1080p) of alpha > 0.02."""
    inside = (_f32(a) > 0.02).astype(np.uint8)
    return cv2.dilate(inside, ellipse(max(1, round(12 * s)))).astype(bool) & ~inside.astype(bool)


def match_clean(B, I, a, s=1.0):
    """v15 perfect: grade the clean plate to the real plate. LTX's clean plate drifts in colour (seen: about -0.04
    blue on grass); the difference measured on the visible ring is spread smoothly (normalized convolution at
    48 px@1080, a local grade), so it fixes drift without copying the ring's detail, and a wrongly rebuilt
    background still shows up in clean_confidence."""
    B, I = _f32(B), _f32(I)
    ring = _ring(a, s)
    if not ring.any():
        return B
    lo = lambda x: cv2.GaussianBlur(x, (0, 0), max(0.5, 2 * s))
    d = (lo(I) - lo(B)) * ring[..., None]
    m = ring.astype(np.float32)[..., None]
    sg = 48 * s
    num, den = _blur_big(d, sg), _blur_big(m, sg)
    off = np.where(den > 0.01, num / np.maximum(den, 1e-6), d[ring].mean(0) if ring.any() else 0)
    return np.clip(B + off, 0, 1).astype(np.float32)


def clean_confidence(B, I, a, others=None, s=1.0):
    """v15 perfect: 0..1 per pixel, how far the clean plate B can be trusted in the edge. Checked where the real
    background is visible (the ring just outside the matte), where B must equal the plate in low frequencies
    (blur 2 px@1080: LTX rebuilds texture such as grass blades differently, and only the colour matters for the
    unmix). The ring's error is pushed inward over the edge (fill_from); 0.02 error or less = trusted, 0.06 or
    more = not (linear). Also 0 where the subject was not removed (B ~ plate inside it). Next to another person (others mask) it is 0: the true background behind the edge
    there is that person, which the clean plate removed."""
    a = _f32(a)
    ring = _ring(a, s)
    lo = lambda x: cv2.GaussianBlur(_f32(x), (0, 0), max(0.5, 2 * s))
    err = np.abs(lo(B) - lo(I)).mean(-1)
    e = fill_from(err[..., None], ring, s, default=[err[ring].mean()])[..., 0] if ring.any() else np.ones_like(a)
    conf = np.clip((0.06 - e) / 0.04, 0, 1).astype(np.float32)
    # Clean Plate fails when people fill the frame: it re-renders them. Then B equals the plate inside the subject
    # too (the ring test still passes). Inside the solid subject B must differ from the plate: 0.03 or less = not
    # removed, 0.08 or more = removed, pushed outward over the edge like the ring error.
    inner = cv2.erode((a >= 0.98).astype(np.uint8), ellipse(max(1, round(6 * s)))).astype(bool)
    if inner.any():
        gone = np.abs(lo(B) - lo(I)).mean(-1)
        g = fill_from(gone[..., None], inner, s, default=[gone[inner].mean()])[..., 0]
        conf *= np.clip((g - 0.03) / 0.05, 0, 1)
    if others is not None:
        near = cv2.dilate((_f32(others) > 0.5).astype(np.uint8), ellipse(max(1, round(24 * s))))
        conf *= 1 - cv2.GaussianBlur(near.astype(np.float32), (0, 0), max(0.5, 4 * s))
        conf[near.astype(bool)] = 0
    return conf


def unmix_colour(I, a, B, F_quick, conf):
    """v15 perfect: F = (I - (1-a) B) / a with the clean plate B, weighted by conf and by alpha: full weight for
    0.25 <= a <= 0.85; toward quick where a is small (the division blows noise up) and where a is near 1 (LTX
    rims run a pixel wide: there the plate is background even at alpha ~1, which only quick's extend fixes)."""
    a = _f32(a)
    a3 = a[..., None]
    Fu = np.clip((_f32(I) - (1 - a3) * _f32(B)) / np.maximum(a3, 1e-3), 0, 1)
    w = (_f32(conf) * _smoothstep(0.05, 0.25, a) * (1 - _smoothstep(0.85, 0.98, a)))[..., None]
    return (w * Fu + (1 - w) * _f32(F_quick)).astype(np.float32)


def _smooth_one(dis, Fs, g, t, radius, t_sigma=1.2, photo_sigma=8.0):
    """Frame t of Fs [n,H,W,3] averaged with its neighbours warped along the optical flow, weighted like
    stabilize (time and photometric error). g: [n,h,w] uint8 plate, may be smaller than Fs (flow scaled up)."""
    H, W = Fs.shape[1:3]
    h, w = g.shape[1:]
    sx, sy = W / w, H / h
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    sgx, sgy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc = Fs[t].astype(np.float32).copy()
    wsum = np.ones((H, W), np.float32)
    for k in range(max(0, t - radius), min(len(Fs), t + radius + 1)):
        if k == t:
            continue
        flow = dis.calc(g[t], g[k], None)
        warped = cv2.remap(g[k], sgx + flow[..., 0], sgy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        err = cv2.GaussianBlur(np.abs(g[t].astype(np.float32) - warped.astype(np.float32)), (0, 0), 2)
        wk = np.exp(-((k - t) ** 2) / (2 * t_sigma ** 2)) * np.exp(-(err / photo_sigma) ** 2)
        if (h, w) != (H, W):
            flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * np.float32([sx, sy])
            wk = cv2.resize(wk, (W, H), interpolation=cv2.INTER_LINEAR)
        acc += wk[..., None] * cv2.remap(Fs[k].astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wsum += wk
    return acc / wsum[..., None]


def smooth_colour(F, a, gray, radius=2, index=None, max_w=1920):
    """Temporal smoothing of F (list of [H,W,3]) inside the edge band (0.002 < a < 0.999), warped along the
    plate's optical flow. gray: [N,H,W] uint8 plate; the flow runs at most max_w wide (4K: half size). Pixels
    outside the band are untouched. index: smooth only that frame and return it (a sliding window over a shot)."""
    Fs = np.stack([_f32(f) for f in F])
    n, h, w = gray.shape
    if w > max_w:
        sw, sh = max_w, max(1, round(h * max_w / w))
        gray = np.stack([cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA) for g in gray])
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = []
    for t in (range(n) if index is None else [index]):
        band = ((_f32(a[t]) > 0.002) & (_f32(a[t]) < 0.999))[..., None]
        out.append(np.where(band, _smooth_one(dis, Fs, gray, t, radius), Fs[t]).astype(np.float32))
    return out if index is None else out[0]


# ---------------------------------------------------------------- QC (measure_matte.py)

def qc(gray, sel, ltx, alpha, others=None):
    """Motion-compensated flicker ratio vs LTX alone + edge detail match vs LTX (1 - sum|alpha - LTX| / sum LTX on
    LTX's soft pixels near the subject: filling the gaps between strands and missing strands both count). Returns dict."""
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
            pred = 0.5 * (cv2.remap(_f32(q[t - 1]), *maps[0], cv2.INTER_LINEAR) + cv2.remap(_f32(q[t + 1]), *maps[1], cv2.INTER_LINEAR))
            acc.append(np.abs(_f32(q[t]) - pred)[m].mean())
    tot = err = 0.0
    for f in range(n):
        T = sel[f] > .5
        lf = _f32(ltx[f])
        O = (lf > .9) & ~T
        dT = cv2.distanceTransform((~T).astype(np.uint8), cv2.DIST_L2, 5)
        dO = cv2.distanceTransform((~O).astype(np.uint8), cv2.DIST_L2, 5)
        hs = (lf > .02) & (lf < .9) & (dT < dO)
        tot += float(lf[hs].sum())
        err += float(np.abs(_f32(alpha[f])[hs] - lf[hs]).sum())
    base = float(np.mean(fl_ltx)) if fl_ltx else 0.0
    return {"flicker_x": float(np.mean(fl_out)) / base if base else 0.0,
            "detail_pct": max(0.0, 1 - err / tot) * 100 if tot else 100.0}
