"""Comfy Pro Alpha Generator math (v12), ported from the CLI pipeline in workflows/matte/*.py.

Pure numpy / OpenCV / torch, no ComfyUI node code. Masks are float32 arrays [N,H,W] in 0-1, plates are uint8
or float arrays [N,H,W,3]. Function names and defaults match the CLI scripts so results stay comparable.
"""
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
