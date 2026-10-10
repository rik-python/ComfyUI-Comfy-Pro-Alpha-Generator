"""Internal stage nodes. The Comfy Pro Alpha Generator node expands into these (plus SAM3 / LTX nodes) at run time.

They are registered so ComfyUI can execute them, but flagged deprecated so they stay out of node search.
Every stage is its own cached node: switching Fast/Best or re-running a preview reuses finished stages.
"""
import json
import os

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

from . import _engine as E
from . import _video as V

CAT = "Comfy Compositing/internal"


def _schema(node_id, inputs, outputs, output_node=False):
    return io.Schema(node_id=node_id, display_name=node_id.replace("ComfyProAlpha_", "CPA "), category=CAT,
                     inputs=inputs, outputs=outputs, is_deprecated=True, is_output_node=output_node)


def _np(t):
    return t.detach().cpu().float().numpy()


def _opt(m):
    return [_np(m)] if m is not None else []


def _h(t):
    """Tensor -> numpy without upcasting (stage outputs are uint8 plates / float16 masks to save RAM)."""
    return t.detach().cpu().numpy()


def _rgb8(plate):
    """IMAGE (float 0-1, or our uint8 plate) -> uint8 numpy [N,H,W,3]."""
    a = _h(plate)
    return a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


def _half(a):
    return torch.from_numpy(np.clip(a, 0, 1).astype(np.float16))


# ------------------------------------------------------------------ plate

def plate_fingerprint(images):
    """Cheap content hash of an IMAGE batch (shape + a strided sample of pixels)."""
    import hashlib
    t = images.detach()
    sample = t[:, ::37, ::41, :].float().cpu().numpy()
    return hashlib.sha1(repr(tuple(t.shape)).encode() + sample.tobytes()).hexdigest()[:16]


class Plate(io.ComfyNode):
    """Plate frames [start, start+count) of the connected images, stored as uint8 (4x less RAM than float32).
    Only our own stage nodes read it; ChunkPad hands LTX a float32 chunk."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Plate",
                       [io.Image.Input("images"), io.Int.Input("start", default=0), io.Int.Input("count", default=0)],
                       [io.Image.Output("plate")])

    @classmethod
    def execute(cls, images, start, count):
        src = images[start:start + count]
        out = torch.empty(src.shape, dtype=torch.uint8)
        for i in range(0, src.shape[0], 16):  # in slices: no full float32 temporary
            out[i:i + 16] = (src[i:i + 16].float().clamp(0, 1) * 255).round().to(torch.uint8)
        return io.NodeOutput(out)


class LoadVideo(io.ComfyNode):
    """VIDEO for SAM3's streaming tracker: the connected images encoded once into a cached proxy mp4."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_LoadVideo", [io.Image.Input("images"), io.Float.Input("fps", default=24.0)],
                       [io.Video.Output("video")])

    @classmethod
    def execute(cls, images, fps):
        from comfy_api.latest import InputImpl
        return io.NodeOutput(InputImpl.VideoFromFile(V.proxy_mp4(images, fps, plate_fingerprint(images))))


class Frame(io.ComfyNode):
    """One frame of the connected images (for the picker's SAM3 detection / part preview)."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Frame", [io.Image.Input("images"), io.Int.Input("index", default=0)],
                       [io.Image.Output("image")])

    @classmethod
    def execute(cls, images, index):
        i = max(0, min(index, images.shape[0] - 1))
        return io.NodeOutput(images[i:i + 1].float().clamp(0, 1))


class PlateProxy(io.ComfyNode):
    """'Load plate' in the picker: small JPEGs of every frame of the connected images for scrubbing."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_PlateProxy", [io.Image.Input("images"), io.Int.Input("width", default=960),
                                                     io.Int.Input("first_frame", default=1)],
                       [], output_node=True)

    @classmethod
    def execute(cls, images, width, first_frame):
        from PIL import Image

        fp = plate_fingerprint(images)
        n, h, w = images.shape[:3]
        sub = os.path.join("comfy_pro_alpha_plate", fp)
        out_dir = os.path.join(folder_paths.get_temp_directory(), sub)
        os.makedirs(out_dir, exist_ok=True)
        size = (min(width, w), round(min(width, w) * h / w))
        pbar = comfy.utils.ProgressBar(n)
        for i in range(n):
            path = os.path.join(out_dir, f"f{i:05d}.jpg")
            if not os.path.isfile(path):
                img = (images[i].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img).resize(size).save(path, quality=85)
            pbar.update(1)
        info = {"frames": int(n), "width": int(w), "height": int(h), "fingerprint": fp, "subfolder": sub.replace("\\", "/"),
                "first_frame": int(first_frame)}
        return io.NodeOutput(ui={"comfy_pro_alpha_plate": [info]})


class TrackMasks(io.ComfyNode):
    """SAM3Propagate output (dict frame -> [obj,H,W]) -> MASK [count,H,W] for the processed range.

    Replaces SAM3VideoOutput, which also renders full-length RGB + visualization batches (~90 s and ~12 GB per
    track); here only the mask is built and only for the frames we use.
    """

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_TrackMasks", [
            io.Custom("SAM3_VIDEO_MASKS").Input("masks"), io.Int.Input("start", default=0),
            io.Int.Input("count", default=1), io.Int.Input("width", default=1920), io.Int.Input("height", default=1080),
        ], [io.Mask.Output("mask")])

    @classmethod
    def execute(cls, masks, start, count, width, height):
        out = torch.zeros(count, height, width, dtype=torch.float16)
        for t in range(count):
            m = masks.get(str(start + t))
            if m is None:
                continue
            m = torch.as_tensor(m).float()
            while m.dim() > 3:
                m = m.flatten(0, 1)
            if m.numel() == 0:  # object lost / left frame: SAM3 returns [0,H,W]
                continue
            if m.dim() == 3:
                m = m.amax(0)
            if m.numel() and m.max() > 1.0:
                m = m / 255.0
            if m.shape[0] in (height, height + 1) and m.shape[1] in (width, width + 1):
                m = m[:height, :width]  # proxy mp4 padded odd sizes to even
            if m.shape != (height, width):
                m = torch.nn.functional.interpolate(m[None, None], size=(height, width), mode="bilinear")[0, 0]
            out[t] = m.clamp(0, 1).half()
        return io.NodeOutput(out)


class MaskMax(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_MaskMax", [io.Mask.Input("a"), io.Mask.Input("b")], [io.Mask.Output("mask")])

    @classmethod
    def execute(cls, a, b):
        return io.NodeOutput(torch.maximum(a, b))


# ------------------------------------------------------------------ text conditioning

EMPTY_PROMPT = os.path.join(os.path.dirname(__file__), "data", "empty_prompt_ltx-2.5.safetensors")


class EmptyPrompt(io.ComfyNode):
    """Alpha-gen runs with an empty prompt, so its conditioning is a constant. It ships in data/ (encoded once with
    the LTX-2.5 gemma4 text encoder), which skips loading the 13 GB text encoder on every machine."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_EmptyPrompt", [], [io.Conditioning.Output("conditioning")])

    @classmethod
    def execute(cls):
        cond = comfy.utils.load_torch_file(EMPTY_PROMPT)["cond"]
        return io.NodeOutput([[cond, {"pooled_output": None, "unprocessed_ltxav_embeds": True}]])


# ------------------------------------------------------------------ selection + ViTMatte + stabilize

_VITMATTE = {}


def _vitmatte(name):
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    if name not in _VITMATTE:
        path = folder_paths.get_full_path("vitmatte", name) if name != "auto-download" else None
        if path is None or not os.path.isdir(path):
            path = _download_vitmatte()
        proc = VitMatteImageProcessor.from_pretrained(path)
        model = VitMatteForImageMatting.from_pretrained(path).to(comfy.model_management.get_torch_device(), torch.float16)
        model.train(False)  # inference mode
        _VITMATTE.clear()
        _VITMATTE[name] = (model, proc)
    return _VITMATTE[name]


def _download_vitmatte():
    from huggingface_hub import snapshot_download

    dest = os.path.join(folder_paths.models_dir, "vitmatte", "vitmatte-base-composition-1k")
    if not os.path.isfile(os.path.join(dest, "config.json")):
        snapshot_download("hustvl/vitmatte-base-composition-1k", local_dir=dest)
    return dest


class Refine(io.ComfyNode):
    """Selection (target + adds - cuts), ONYX-style ViTMatte edge, optical-flow stabilization."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Refine", [
            io.Image.Input("plate"), io.Mask.Input("target"),
            io.Mask.Input("adds", optional=True), io.Mask.Input("cuts", optional=True),
            io.Mask.Input("others", optional=True),
            io.String.Input("vitmatte", default="auto-download"),
            io.Int.Input("erode", default=10), io.Int.Input("dilate", default=15), io.Int.Input("radius", default=2),
        ], [io.Mask.Output("selection"), io.Mask.Output("refined"), io.Mask.Output("others")])

    @classmethod
    def execute(cls, plate, target, vitmatte, erode, dilate, radius, adds=None, cuts=None, others=None):
        n = plate.shape[0]
        rgb = _rgb8(plate)
        sel, oth = E.build_selection(_np(target), _opt(adds), _opt(cuts), _opt(others))

        model, proc = _vitmatte(vitmatte)
        pbar = comfy.utils.ProgressBar(2 * n)
        vit = np.empty_like(sel)
        for f in range(n):
            comfy.model_management.throw_exception_if_processing_interrupted()
            vit[f] = E.vitmatte_frame(model, proc, rgb[f], sel[f], erode, dilate)
            pbar.update(1)
        _VITMATTE.clear()
        comfy.model_management.soft_empty_cache()

        sel, vit, oth = E.stabilize([sel, vit, oth], E.to_gray(rgb), radius=radius,
                                    progress=lambda i, t: pbar.update(1))
        # selection / others stay float32: they shape the isolated plate, and LTX amplifies even 1-level rounding
        # there into a different (equally valid) sample. The ViTMatte edge only feeds the final blend: half is exact enough.
        f32 = lambda a: torch.from_numpy(np.clip(a, 0, 1).astype(np.float32))
        return io.NodeOutput(f32(sel), _half(vit), f32(oth))


# ------------------------------------------------------------------ isolated plate

class Isolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Isolate", [
            io.Image.Input("plate"), io.Mask.Input("selection"), io.Mask.Input("others"),
            io.Int.Input("grow", default=6), io.Float.Input("blur", default=6.0),
            io.Int.Input("start", default=0, optional=True),
        ], [io.Image.Output("isolated")])

    @classmethod
    def execute(cls, plate, selection, others, grow, blur, start=0):
        # plate = the full-precision upstream frames (from `start`), read one frame at a time: the inpaint fill
        # depends on the unrounded colours, and LTX turns a 1-level change there into a different sample.
        sel, oth = _h(selection), _h(others)
        n = sel.shape[0]
        out = np.empty((n,) + tuple(plate.shape[1:]), np.uint8)
        pbar = comfy.utils.ProgressBar(n)
        for f in range(n):
            fr = plate[start + f]
            rgb = fr.cpu().numpy().astype(np.float32) if fr.dtype == torch.uint8 else fr.float().clamp(0, 1).cpu().numpy() * 255.0
            iso = E.isolate_frame(rgb, E._f32(sel[f]), E._f32(oth[f]), grow, blur)
            out[f] = np.clip(iso, 0, 255).round().astype(np.uint8)  # 8 bit, like the CLI's mp4 plate
            pbar.update(1)
        return io.NodeOutput(torch.from_numpy(out))


# ------------------------------------------------------------------ LTX chunking

class ChunkPad(io.ComfyNode):
    """Slice one LTX chunk and pad it: H/W to a multiple of 32 (replicate edge), length to 8k+1 (repeat last)."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_ChunkPad", [
            io.Image.Input("images"), io.Int.Input("start", default=0), io.Int.Input("length", default=49),
            io.Int.Input("end", default=-1),
        ], [io.Image.Output("chunk")])

    @classmethod
    def execute(cls, images, start, length, end=-1):
        stop = start + length if end < 0 else min(start + length, end)  # never past the processed range
        c = images[start:stop]
        c = c.float() / 255.0 if c.dtype == torch.uint8 else c.float().clamp(0, 1)
        if c.shape[0] < length:
            c = torch.cat([c, c[-1:].repeat(length - c.shape[0], 1, 1, 1)])
        h, w = c.shape[1:3]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            c = torch.nn.functional.pad(c.permute(0, 3, 1, 2), (0, pw, 0, ph), mode="replicate").permute(0, 2, 3, 1)
        return io.NodeOutput(c)


class DecodeAlpha(io.ComfyNode):
    """VAE-decode one LTX chunk (tiled, same settings as before) and keep only its alpha, as float16.
    Fused so the full-colour float32 decode (about 1.2 GB per 1080p chunk) is never cached or held for Stitch."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_DecodeAlpha", [io.Latent.Input("samples"), io.Vae.Input("vae")],
                       [io.Mask.Output("alpha")])

    @classmethod
    def execute(cls, samples, vae):
        import nodes

        img = nodes.VAEDecodeTiled().decode(vae, samples, 512, 64, 64, 8)[0]
        return io.NodeOutput(img.float().mean(-1).half().cpu())


class Stitch(io.ComfyNode):
    """Cross-fade decoded LTX chunks (triangular weights) into one alpha MASK of the processed range."""
    MAX = 24

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Stitch",
                       [io.String.Input("plan")] + [io.Mask.Input(f"chunk_{i}", optional=True) for i in range(cls.MAX)],
                       [io.Mask.Output("alpha")])

    @classmethod
    def execute(cls, plan, **chunks):
        p = json.loads(plan)
        n, h, w, L, starts = p["n"], p["h"], p["w"], p["chunk"], p["starts"]
        acc = torch.zeros(n, h, w)
        wsum = torch.zeros(n, 1, 1)
        for i, s in enumerate(starts):
            img = chunks[f"chunk_{i}"][:, :h, :w].float().cpu()
            if img.dim() == 4:  # an IMAGE chunk (older graph): average the channels
                img = img.mean(-1)
            for k in range(min(L, n - s)):
                wt = E.chunk_weight(i, starts, L, k)
                acc[s + k] += img[k] * wt
                wsum[s + k] += wt
        return io.NodeOutput((acc / wsum).clamp(0, 1).half())


# ------------------------------------------------------------------ final

def _save(alpha, prefix, fmt, first_frame):
    import cv2

    out_dir = os.path.join(folder_paths.get_output_directory(), prefix)
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "exr":
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    for i, a in enumerate(alpha):
        path = os.path.join(out_dir, f"alpha_{first_frame + i:05d}.{'exr' if fmt == 'exr' else 'png'}")
        if fmt == "exr":
            ok = cv2.imwrite(path, a.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                                                         cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
        else:
            ok = cv2.imwrite(path, (np.clip(a, 0, 1) * 65535).round().astype(np.uint16))
        if not ok:
            raise RuntimeError(f"could not write {path}" + (" (OpenCV EXR disabled: start ComfyUI with "
                                                             "OPENCV_IO_ENABLE_OPENEXR=1 or use png16)" if fmt == "exr" else ""))
    return out_dir


def _previews(plate, alpha, every):
    """Small JPEG pairs (plate, comp over grey) in temp/ for the in-node compare viewer."""
    from PIL import Image

    tmp = folder_paths.get_temp_directory()
    sub = "comfy_pro_alpha_preview"
    os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    tag = os.urandom(4).hex()
    items = []
    h, w = alpha.shape[1:]
    size = (min(960, w), int(min(960, w) * h / w))
    for f in range(0, len(alpha), every):
        rgb = plate[f].astype(np.float32) / 255.0 if plate.dtype == np.uint8 else plate[f]
        a = alpha[f][..., None]
        comp = rgb * a + 0.5 * (1 - a)
        names = []
        for kind, img in (("plate", rgb), ("comp", comp), ("alpha", np.repeat(a, 3, -1))):
            name = f"sm_{tag}_{f:05d}_{kind}.jpg"
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(size).save(os.path.join(tmp, sub, name), quality=85)
            names.append({"filename": name, "subfolder": sub, "type": "temp"})
        items.append({"frame": f, "plate": names[0], "comp": names[1], "alpha": names[2]})
    return items


class Final(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Final", [
            io.Image.Input("plate"), io.Mask.Input("ltx"), io.Mask.Input("ltx_isolated", optional=True),
            io.Mask.Input("selection"), io.Mask.Input("refined"), io.Mask.Input("others"),
            io.Int.Input("near_px", default=15), io.String.Input("filename_prefix", default="comfy_pro_alpha/shot"),
            io.Combo.Input("format", options=["png16", "exr"]), io.Int.Input("first_frame", default=1),
            io.Boolean.Input("save", default=True), io.Boolean.Input("qc", default=True),
            io.Boolean.Input("comp", default=True),
        ], [io.Mask.Output("alpha"), io.Image.Output("comp")], output_node=True)

    @classmethod
    def execute(cls, plate, ltx, selection, refined, others, near_px, filename_prefix, format, first_frame,
                save, qc, comp=True, ltx_isolated=None):
        p = _rgb8(plate)
        L, S, R, O = _h(ltx), _h(selection), _h(refined), _h(others)
        I = _h(ltx_isolated) if ltx_isolated is not None else None
        n, f32 = len(p), E._f32
        alpha = np.empty(L.shape, np.float32)
        for f in range(n):  # frame by frame: no full-shot v7 / v11 stacks
            v7 = E.hybrid(f32(L[f]), f32(S[f]), f32(R[f]))
            alpha[f] = v7 if I is None else E.v12_blend(v7, E.level(f32(I[f]), f32(S[f]), f32(O[f]))[0],
                                                        f32(S[f]), f32(O[f]), near_px)
        mode = "v12 (best)" if I is not None else "v7 (fast)"
        # LTX alpha-gen only mattes what it sees as the main subject: a background car or a prop can come out
        # empty. Fall back to the ViTMatte edge wherever LTX does not cover the selection.
        ltx_seen = np.maximum(L, I) if I is not None else L
        alpha, cover = E.ltx_fallback(alpha, ltx_seen, S, R, out=alpha)
        np.clip(alpha, 0, 1, out=alpha)

        out_dir = _save(alpha, filename_prefix, format, first_frame) if save else ""
        score = E.qc(E.to_gray(p), S, L, alpha) if qc else {}
        if comp:  # only built when the node's comp output is wired (a full 1080p shot is several GB)
            comp_t = torch.empty(p.shape, dtype=torch.float32)
            for f in range(n):
                a = alpha[f][..., None]
                comp_t[f] = torch.from_numpy(p[f].astype(np.float32) / 255.0 * a + 0.5 * (1 - a))
        else:
            comp_t = torch.zeros(1, 64, 64, 3)
        info = {"mode": mode, "frames": n, "first_frame": first_frame, "out_dir": out_dir, "format": format,
                "ltx_cover": float(np.mean(cover) * 100), **{k: float(v) for k, v in score.items()}}
        ui_out = {"pro_alpha": [info], "previews": _previews(p, alpha, max(1, n // 48))}
        return io.NodeOutput(torch.from_numpy(alpha), comp_t, ui=ui_out)


# ------------------------------------------------------------------ detection report (for the picker UI)

class DetectReport(io.ComfyNode):
    """Turns SAM3 text detections (any subject) on one frame into boxes + mask PNGs the picker draws as overlays."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_DetectReport",
                       [io.Mask.Input("masks"), io.Int.Input("frame", default=0), io.String.Input("token", default="")],
                       [], output_node=True)

    @classmethod
    def execute(cls, masks, frame, token):
        from PIL import Image

        m = _np(masks)
        if m.ndim == 2:
            m = m[None]
        tmp = folder_paths.get_temp_directory()
        sub = "comfy_pro_alpha_detect"
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        subjects = []
        h, w = m.shape[1:]
        for i, mk in enumerate(m):
            ys, xs = np.where(mk > 0.5)
            if len(ys) < 50:
                continue
            x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
            name = f"det_{token}_{i}.png"
            Image.fromarray(((mk > 0.5) * 255).astype(np.uint8), "L").save(os.path.join(tmp, sub, name))
            subjects.append({"id": len(subjects), "box": [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0],
                           "area": float((mk > 0.5).mean()),
                           "mask": {"filename": name, "subfolder": sub, "type": "temp"}})
        return io.NodeOutput(ui={"comfy_pro_alpha_detect": [{"frame": frame, "token": token, "subjects": subjects}]})


class PartReport(io.ComfyNode):
    """Single-frame mask preview for a cut/add click (SAM3 point segmentation)."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_PartReport",
                       [io.Mask.Input("mask"), io.String.Input("token", default="")], [], output_node=True)

    @classmethod
    def execute(cls, mask, token):
        from PIL import Image

        m = _np(mask)
        m = m[0] if m.ndim == 3 else m
        tmp = folder_paths.get_temp_directory()
        sub = "comfy_pro_alpha_detect"
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        name = f"part_{token}.png"
        Image.fromarray(((m > 0.5) * 255).astype(np.uint8), "L").save(os.path.join(tmp, sub, name))
        return io.NodeOutput(ui={"comfy_pro_alpha_part": [{"token": token,
                                                          "mask": {"filename": name, "subfolder": sub, "type": "temp"}}]})


NODES = [Plate, LoadVideo, Frame, PlateProxy, TrackMasks, MaskMax, EmptyPrompt, Refine, Isolate, ChunkPad, DecodeAlpha, Stitch, Final, DetectReport, PartReport]
