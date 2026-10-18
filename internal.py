"""Internal stage nodes. The Comfy Pro Alpha Generator node expands into these (plus SAM3 / LTX nodes) at run time.

They are registered so ComfyUI can execute them, but flagged deprecated so they stay out of node search.
Every stage is its own cached node: switching Fast/Best or re-running a preview reuses finished stages.
"""
import json
import os

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

from . import _engine as E
from . import _video as V

CAT = "Comfy Compositing/internal"


def _schema(node_id, inputs, outputs, output_node=False, **kw):
    return io.Schema(node_id=node_id, display_name=node_id.replace("ComfyProAlpha_", "CPA "), category=CAT,
                     inputs=inputs, outputs=outputs, is_deprecated=True, is_output_node=output_node, **kw)


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


# ------------------------------------------------------------------ depth (v13 front/back order)

DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0 (Base/Large are CC-BY-NC: not usable here)


class Depth(io.ComfyNode):
    """v13: relative depth per frame (bright = near), so the overlap pixels SAM gives to both people go to whoever
    is in front. Depth Anything V2 Small, downloaded once (~100 MB) to models/depthanything/."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Depth", [io.Image.Input("plate")], [io.Mask.Output("depth")])

    @classmethod
    def execute(cls, plate):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        path = os.path.join(folder_paths.models_dir, "depthanything", DEPTH_REPO.split("/")[1])
        if not os.path.isfile(os.path.join(path, "config.json")):
            from huggingface_hub import snapshot_download
            snapshot_download(DEPTH_REPO, local_dir=path)
        dev = comfy.model_management.get_torch_device()
        proc = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForDepthEstimation.from_pretrained(path).to(dev, torch.float16)
        model.train(False)  # inference mode
        rgb = _rgb8(plate)
        n, h, w = rgb.shape[:3]
        out = torch.empty(n, h, w, dtype=torch.float16)
        pbar = comfy.utils.ProgressBar(n)
        for f in range(n):
            comfy.model_management.throw_exception_if_processing_interrupted()
            inp = proc(images=rgb[f], return_tensors="pt").to(dev, torch.float16)
            with torch.no_grad():
                d = model(**inp).predicted_depth[:1].float()
            d = torch.nn.functional.interpolate(d[None], size=(h, w), mode="bicubic")[0, 0].cpu()
            out[f] = ((d - d.min()) / (d.max() - d.min()).clamp(min=1e-6)).half()
            pbar.update(1)
        del model
        comfy.model_management.soft_empty_cache()
        return io.NodeOutput(out)


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
        return io.NodeOutput(_pad_chunk(images[start:stop], length))


def _pad_chunk(c, length):
    """Float 0-1, length padded to `length` (repeat last frame), H/W padded to a multiple of 32 (replicate edge)."""
    c = c.float() / 255.0 if c.dtype == torch.uint8 else c.float().clamp(0, 1)
    if c.shape[0] < length:
        c = torch.cat([c, c[-1:].repeat(length - c.shape[0], 1, 1, 1)])
    h, w = c.shape[1:3]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        c = torch.nn.functional.pad(c.permute(0, 3, 1, 2), (0, pw, 0, ph), mode="replicate").permute(0, 2, 3, 1)
    return c


class AttnChunk(io.ComfyNode):
    """v13: one LTX chunk of a subject mask as the IC-LoRA guide's attention_mask: grown + feathered, then padded
    like ChunkPad (H/W to a multiple of 32, length to the chunk) so it lines up with the guide frames."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_AttnChunk", [
            io.Mask.Input("mask"), io.Int.Input("start", default=0), io.Int.Input("length", default=49),
            io.Int.Input("end", default=-1), io.Int.Input("grow", default=20), io.Float.Input("blur", default=10.0),
        ], [io.Mask.Output("attention_mask")])

    @classmethod
    def execute(cls, mask, start, length, end=-1, grow=20, blur=10.0):
        stop = start + length if end < 0 else min(start + length, end)
        src = _h(mask[start:stop])
        out = np.stack([E.attn_mask(m, grow, blur) for m in src])
        if len(out) < length:
            out = np.concatenate([out, np.repeat(out[-1:], length - len(out), 0)])
        h, w = out.shape[1:]
        out = np.pad(out, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="edge")
        return io.NodeOutput(torch.from_numpy(np.clip(out, 0, 1).astype(np.float32)))


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


# ------------------------------------------------------------------ v14 native-resolution edges

class Proxy(io.ComfyNode):
    """v14: the plate scaled to fit inside 1920x1088 (area filter, float32 like the upstream frames): the whole
    pipeline runs on it, the native-resolution tiles only refine the edges."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_Proxy", [io.Image.Input("images")], [io.Image.Output("proxy")])

    @classmethod
    def execute(cls, images):
        import cv2

        n, h, w = images.shape[:3]
        pw, ph = E.proxy_size(w, h)
        out = torch.empty((n, ph, pw, images.shape[3]), dtype=torch.float32)
        for i in range(0, n, 16):  # in slices: no full float32 copy of the native batch
            for j, fr in enumerate(images[i:i + 16].float().clamp(0, 1).cpu().numpy()):
                out[i + j] = torch.from_numpy(cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA))
        return io.NodeOutput(out)


class NativeIsolate(io.ComfyNode):
    """v14: the native plate with every other subject painted out, the proxy masks scaled up frame by frame
    (fused: no native-resolution mask stacks are cached). uint8, like Isolate."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_NativeIsolate", [
            io.Image.Input("plate"), io.Mask.Input("selection"), io.Mask.Input("others"),
            io.Int.Input("grow", default=12), io.Float.Input("blur", default=12.0), io.Int.Input("start", default=0),
        ], [io.Image.Output("isolated")])

    @classmethod
    def execute(cls, plate, selection, others, grow, blur, start=0):
        import cv2

        sel, oth = _h(selection), _h(others)
        n, (H, W) = sel.shape[0], plate.shape[1:3]
        out = np.empty((n, H, W, plate.shape[3]), np.uint8)
        pbar = comfy.utils.ProgressBar(n)
        for f in range(n):
            fr = plate[start + f]
            rgb = fr.cpu().numpy().astype(np.float32) if fr.dtype == torch.uint8 else fr.float().clamp(0, 1).cpu().numpy() * 255.0
            up = lambda m: cv2.resize(E._f32(m), (W, H), interpolation=cv2.INTER_LINEAR)
            iso = E.isolate_frame(rgb, up(sel[f]), up(oth[f]), grow, blur)
            out[f] = np.clip(iso, 0, 255).round().astype(np.uint8)
            pbar.update(1)
        return io.NodeOutput(torch.from_numpy(out))


class TileGuide(io.ComfyNode):
    """v14: one LTX chunk of one native-resolution tile as the IC-LoRA guide (LTXAddVideoICLoRAGuide, same
    settings as the 1080p passes). Fused: the float32 tile chunk (about 1.2 GB) is never a cached output."""

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_TileGuide", [
            io.Image.Input("images"), io.Int.Input("x", default=0), io.Int.Input("y", default=0),
            io.Int.Input("width", default=1920), io.Int.Input("height", default=1088),
            io.Int.Input("start", default=0), io.Int.Input("length", default=49), io.Int.Input("end", default=-1),
            io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Vae.Input("vae"),
            io.Latent.Input("latent"), io.Float.Input("latent_downscale_factor", default=1.0),
        ], [io.Conditioning.Output("positive"), io.Conditioning.Output("negative"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, images, x, y, width, height, start, length, end, positive, negative, vae, latent,
                latent_downscale_factor):
        import nodes

        stop = start + length if end < 0 else min(start + length, end)
        img = _pad_chunk(images[start:stop, y:y + height, x:x + width], length)
        guide = nodes.NODE_CLASS_MAPPINGS.get("LTXAddVideoICLoRAGuide")
        if guide is None:
            raise RuntimeError("Native-resolution edges need ComfyUI's LTXAddVideoICLoRAGuide node: update ComfyUI "
                               "(or switch native-resolution edges off).")
        kw = dict(positive=positive, negative=negative, vae=vae, latent=latent, image=img, frame_idx=0, strength=1.0,
                  latent_downscale_factor=latent_downscale_factor, crop="disabled", use_tiled_encode=False,
                  tile_size=256, tile_overlap=64)
        out = guide.execute(**kw) if hasattr(guide, "execute") else getattr(guide(), guide.FUNCTION)(**kw)
        res = tuple(getattr(out, "result", out) or ())  # V3 NodeOutput or a V1 tuple
        if len(res) < 3:
            raise RuntimeError(f"LTXAddVideoICLoRAGuide returned {len(res)} outputs, expected positive, negative, "
                               "latent: this ComfyUI version changed the node; please report it.")
        return io.NodeOutput(*res[:3])


class DecodeStitch(io.ComfyNode):
    """v14: decode a tile's LTX chunks one at a time and cross-fade them (DecodeAlpha + Stitch fused: no decoded
    chunk is a cached output)."""
    MAX = 24

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_DecodeStitch",
                       [io.String.Input("plan"), io.Vae.Input("vae")]
                       + [io.Latent.Input(f"chunk_{i}", optional=True) for i in range(cls.MAX)],
                       [io.Mask.Output("alpha")])

    @classmethod
    def execute(cls, plan, vae, **chunks):
        import nodes

        p = json.loads(plan)
        n, h, w, L, starts = p["n"], p["h"], p["w"], p["chunk"], p["starts"]
        acc = torch.zeros(n, h, w)
        wsum = torch.zeros(n, 1, 1)
        for i, s in enumerate(starts):
            img = nodes.VAEDecodeTiled().decode(vae, chunks[f"chunk_{i}"], 512, 64, 64, 8)[0]
            img = img.float().mean(-1).half().cpu()[:, :h, :w].float()  # as DecodeAlpha -> Stitch
            for k in range(min(L, n - s)):
                wt = E.chunk_weight(i, starts, L, k)
                acc[s + k] += img[k] * wt
                wsum[s + k] += wt
            del img
        return io.NodeOutput((acc / wsum).clamp(0, 1).half())


class DecodeStitchRGB(io.ComfyNode):
    """v15 perfect: decode the clean plate's LTX chunks one at a time to RGB and cross-fade them (as Stitch), kept
    as uint8 like the plate (a 1080p shot in float32 would be several GB). Frames go to uint8 as soon as no later
    chunk touches them, so only one chunk's overlap is ever held in float."""
    MAX = 24

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_DecodeStitchRGB",
                       [io.String.Input("plan"), io.Vae.Input("vae")]
                       + [io.Latent.Input(f"chunk_{i}", optional=True) for i in range(cls.MAX)],
                       [io.Image.Output("clean")])

    @classmethod
    def execute(cls, plan, vae, **chunks):
        import nodes

        p = json.loads(plan)
        n, h, w, L, starts = p["n"], p["h"], p["w"], p["chunk"], p["starts"]
        out = torch.empty((n, h, w, 3), dtype=torch.uint8)
        acc, wsum = {}, {}  # frame -> float sum / weight, only for frames a later chunk still adds to
        for i, s in enumerate(starts):
            img = nodes.VAEDecodeTiled().decode(vae, chunks[f"chunk_{i}"], 512, 64, 64, 8)[0]
            img = img.float().cpu()[:, :h, :w]
            for k in range(min(L, n - s)):
                wt = E.chunk_weight(i, starts, L, k)
                f = s + k
                acc[f] = img[k] * wt if f not in acc else acc[f].add_(img[k], alpha=wt)
                wsum[f] = wsum.get(f, 0.0) + wt
            del img
            done = n if i == len(starts) - 1 else starts[i + 1]
            for f in [f for f in acc if f < done]:
                out[f] = acc.pop(f).div_(wsum.pop(f)).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        return io.NodeOutput(out)


class TileMerge(io.ComfyNode):
    """v14: feather-blend the rendered tiles into native-resolution frames, plus where they cover the plate."""
    MAX = 64

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_TileMerge",
                       [io.String.Input("plan")] + [io.Mask.Input(f"tile_{i}", optional=True) for i in range(cls.MAX)],
                       [io.Mask.Output("tiles"), io.Mask.Output("coverage")])

    @classmethod
    def execute(cls, plan, **tiles):
        p = json.loads(plan)
        n, W, H, grid, used = p["n"], p["W"], p["H"], [tuple(g) for g in p["grid"]], p["used"]
        feathers = [E.tile_feather(grid, i, (W, H)) for i in range(len(grid))]
        src = [_h(tiles[f"tile_{k}"]) for k in range(len(used))]
        alpha = np.zeros((n, H, W), np.float16)
        cover = np.zeros((1, H, W), np.float16)  # same for every frame: the tile set is chosen for the whole range
        for f in range(n):
            alpha[f], c = E.stitch_tiles([t[f] for t in src], grid, used, (W, H), feathers)
            if f == 0:
                cover[0] = c
        return io.NodeOutput(_half(alpha), _half(cover))


class NativeEdges(io.ComfyNode):
    """v14: once the proxy selection exists, render one isolated LTX pass per native-resolution tile that touches
    the subject's edge (others painted out at native resolution), then feather the tiles together.
    Expands at run time: which tiles are needed is only known after SAM3 has tracked the subject."""

    @classmethod
    def define_schema(cls):
        # Everything the tiles depend on is a real input, even though only the links (in `config`) are used: ComfyUI
        # keys its cache on real inputs, so a new seed, model, LoRA, VAE, encoder or fps must re-render the tiles.
        deps = [io.Image.Input("plate", optional=True), io.Mask.Input("others", optional=True),
                io.Model.Input("model", optional=True), io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True), io.Vae.Input("vae", optional=True),
                io.Latent.Input("audio_latent", optional=True), io.Float.Input("lora_scale", optional=True),
                io.Noise.Input("noise", optional=True), io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True)]
        return _schema("ComfyProAlpha_NativeEdges", [io.Mask.Input("selection"), io.String.Input("config")] + deps,
                       [io.Mask.Output("tiles"), io.Mask.Output("coverage"), io.Int.Output("count")],
                       enable_expand=True)

    @classmethod
    def execute(cls, selection, config, **deps):
        from comfy_execution.graph_utils import GraphBuilder

        from .node import _ltx_chunks

        c = json.loads(config)
        W, H, count, start, L, starts, lk = c["W"], c["H"], c["count"], c["start"], c["L"], c["starts"], c["links"]
        sel = _h(selection)
        grid = E.tile_grid(W, H)
        used = E.select_tiles(sel, grid, (W, H))
        print(f"[ComfyProAlpha] native edges: {len(used)} of {len(grid)} tiles at {W}x{H}")
        if not used:  # SAM3 lost the subject in the whole range: Final keeps the scaled-up proxy alpha
            z = torch.zeros((0, 1, 1), dtype=torch.float16)
            return io.NodeOutput(z, z, 0)
        if len(used) > TileMerge.MAX:
            raise ValueError(f"A {W}x{H} plate needs {len(used)} native tiles; native-resolution edges handle up to "
                             f"{TileMerge.MAX} (plates up to about 10K wide). Switch native-resolution edges off, or "
                             "crop the plate.")
        s = W / sel.shape[2]
        g = GraphBuilder()
        iso = g.node("ComfyProAlpha_NativeIsolate", plate=lk["plate"], start=start, selection=lk["selection"],
                     others=lk["others"], grow=round(6 * s), blur=6.0 * s)
        tiles = {}
        for k, i in enumerate(used):
            x, y, tw, th = grid[i]
            lat = g.node("EmptyLTXVLatentVideo", width=tw + (-tw) % 32, height=th + (-th) % 32, length=L, batch_size=1)
            plan = json.dumps({"n": count, "h": th, "w": tw, "chunk": L, "starts": starts})
            tiles[f"tile_{k}"] = _ltx_chunks(g, {**lk["stack"], "latent": lat.out(0)}, iso.out(0), 0, count, starts, L,
                                             plan, tile=(x, y, tw, th))
        merge = g.node("ComfyProAlpha_TileMerge", plan=json.dumps({"n": count, "W": W, "H": H, "grid": grid,
                                                                   "used": used}), **tiles)
        return io.NodeOutput(merge.out(0), merge.out(1), len(used), expand=g.finalize())


# ------------------------------------------------------------------ final

def _native_alpha(alpha, native_hw, tiles, coverage, others):
    """v14: the proxy alpha scaled up to the plate's native size; with tiles, their edges merged in away from the
    other subjects (float32, like the 1080p path). others: the proxy's others mask. coverage: one frame (or one per
    frame); empty tiles (nothing rendered) = no tiles."""
    import cv2

    H, W = native_hw
    n, _, pw = alpha.shape
    T = _h(tiles) if tiles is not None and len(tiles) else None
    C = _h(coverage) if T is not None else None
    out = np.empty((n, H, W), np.float32)
    for f in range(n):
        a = np.clip(cv2.resize(alpha[f], (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
        if T is not None:
            o = cv2.resize(E._f32(others[f]), (W, H), interpolation=cv2.INTER_LINEAR)
            a = E.merge_native(a, T[f], C[0] if len(C) == 1 else C[f], W / pw, o)
        out[f] = a
    return out


def _out_dir(prefix):
    """filename_prefix -> folder under ComfyUI/output (a prefix like ../x must not write outside it, as SaveImage)."""
    base = os.path.realpath(folder_paths.get_output_directory())
    out = os.path.realpath(os.path.join(base, prefix))
    if os.path.commonpath([base, out]) != base:
        raise ValueError(f"filename_prefix '{prefix}' points outside the ComfyUI output folder")
    os.makedirs(out, exist_ok=True)
    return out


def _save(alpha, prefix, fmt, first_frame):
    import cv2

    out_dir = _out_dir(prefix)
    if fmt == "exr":
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    for i, a in enumerate(alpha):
        a = E._f32(a)  # v14 native frames are float16: 65535 overflows it
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


def _save_rgba(F, a, out_dir, fmt, frame):
    """v15: one frame of the clean colour. EXR: half float, scene-linear, premultiplied. PNG: 16-bit sRGB, straight
    alpha (the PNG convention). F: [H,W,3] linear; a: [H,W]."""
    import cv2

    a = E._f32(a)
    path = os.path.join(out_dir, f"rgba_{frame:05d}.{'exr' if fmt == 'exr' else 'png'}")
    if fmt == "exr":
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        rgba = np.dstack([np.clip(F, 0, 1) * a[..., None], a])[..., [2, 1, 0, 3]].astype(np.float32)
        ok = cv2.imwrite(path, rgba, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                                      cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
    else:
        rgba = np.dstack([E.linear_to_srgb(F), np.clip(a, 0, 1)])[..., [2, 1, 0, 3]]
        ok = cv2.imwrite(path, (rgba * 65535).round().astype(np.uint16))
    if not ok:
        raise RuntimeError(f"could not write {path}" + (" (OpenCV EXR disabled: start ComfyUI with "
                                                         "OPENCV_IO_ENABLE_OPENEXR=1 or use png16)" if fmt == "exr" else ""))


def _previews(plate, alpha, every, clean=None):
    """Small JPEG pairs (plate, comp over grey) in temp/ for the in-node compare viewer. clean (v15): frame ->
    linear clean colour already at preview size; the comp is then made from it, over grey, in linear light."""
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
        if clean is not None:  # v15: work at preview size (a 4K frame list would be GBs)
            import cv2
            rgb = cv2.resize(E._f32(rgb), size, interpolation=cv2.INTER_AREA)
            a = cv2.resize(E._f32(alpha[f]), size, interpolation=cv2.INTER_AREA)[..., None]
            grey = E.srgb_to_linear(np.float32(0.5))
            comp = E.linear_to_srgb(clean[f] * a + grey * (1 - a))
        else:
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
            io.Mask.Input("attn_target", optional=True), io.Mask.Input("attn_others", optional=True),
            io.Mask.Input("depth", optional=True), io.Boolean.Input("hair_guide", default=False, optional=True),
            io.Int.Input("hair_radius", default=4, optional=True), io.Int.Input("occluder_reach", default=20, optional=True),
            io.Int.Input("limb_cap", default=90, optional=True), io.Int.Input("front_smoothing", default=3, optional=True),
            io.Mask.Input("selection"), io.Mask.Input("refined"), io.Mask.Input("others"),
            io.Int.Input("near_px", default=15), io.String.Input("filename_prefix", default="comfy_pro_alpha/shot"),
            io.Combo.Input("format", options=["png16", "exr"]), io.Int.Input("first_frame", default=1),
            io.Boolean.Input("save", default=True), io.Boolean.Input("qc", default=True),
            io.Boolean.Input("comp", default=True),
            io.Image.Input("native_plate", optional=True), io.Int.Input("native_start", default=0, optional=True),
            io.Mask.Input("tiles", optional=True), io.Mask.Input("coverage", optional=True),
            io.Int.Input("tile_count", default=0, optional=True),
        ], [io.Mask.Output("alpha"), io.Image.Output("comp"), io.String.Output("info")], output_node=True)

    @classmethod
    def execute(cls, plate, ltx, selection, refined, others, near_px, filename_prefix, format, first_frame,
                save, qc, comp=True, ltx_isolated=None, attn_target=None, attn_others=None, depth=None,
                hair_guide=False, hair_radius=4, occluder_reach=20, limb_cap=90, front_smoothing=3,
                native_plate=None, native_start=0, tiles=None, coverage=None, tile_count=0):
        p = _rgb8(plate)
        L, S, R, O = _h(ltx), _h(selection), _h(refined), _h(others)
        I = _h(ltx_isolated) if ltx_isolated is not None else None
        n, f32 = len(p), E._f32
        alpha = np.empty(L.shape, np.float32)
        if attn_target is not None and attn_others is not None:  # v13: attention passes + front/back order
            E.v13_overlap(L, _h(attn_target), _h(attn_others), S, O, out=alpha,
                          depth=None if depth is None else _h(depth), gray=E.to_gray(p) if hair_guide else None,
                          guide_r=hair_radius, grow=occluder_reach, far=limb_cap, near=max(0, limb_cap - 40),
                          tsmooth=front_smoothing)
            mode = "v13 (overlap)"
        else:
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

        # v14: a plate bigger than 1920x1088 was matted on a proxy: the output is at the plate's own size
        native = None if native_plate is None else _native_alpha(alpha, tuple(native_plate.shape[1:3]), tiles,
                                                                  coverage, O)
        out = alpha if native is None else native
        out_dir = _save(out, filename_prefix, format, first_frame) if save else ""
        score = E.qc(E.to_gray(p), S, L, alpha) if qc else {}  # QC on the proxy (as for any 1080p plate)
        if comp:  # only built when the node's comp output is wired (a full 1080p shot is several GB)
            comp_t = torch.empty(out.shape + (3,), dtype=torch.float32)
            for f in range(n):
                a = E._f32(out[f])[..., None]
                rgb = (p[f].astype(np.float32) / 255.0 if native is None else
                       native_plate[native_start + f].float().clamp(0, 1).cpu().numpy())
                comp_t[f] = torch.from_numpy(rgb * a + 0.5 * (1 - a))
        else:
            comp_t = torch.zeros(1, 64, 64, 3)
        info = {"mode": mode, "frames": n, "first_frame": first_frame, "out_dir": out_dir, "format": format,
                "ltx_cover": float(np.mean(cover) * 100), **{k: float(v) for k, v in score.items()}}
        if native is not None:
            info.update(native=f"{native.shape[2]}x{native.shape[1]}", native_edges=tiles is not None,
                        tiles=int(tile_count))
        # previews stay proxy-sized (they are <= 960 px wide anyway)
        ui_out = {"pro_alpha": [info], "previews": _previews(p, alpha, max(1, n // 48))}
        return io.NodeOutput(torch.from_numpy(out), comp_t, json.dumps(info), ui=ui_out)


EDGE_NOTE = "perfect: Clean Plate LoRA not found, ran quick"


class EdgeColour(io.ComfyNode):
    """v15: the subject's own colour in the soft edges (quick: Blur-Fusion + texture extend; perfect: clean plate
    unmix), in linear light. Writes rgba_ files next to the alpha and the clean premultiplied comp. Never changes
    the alpha. Its own stage, so switching quick/perfect does not re-run Final."""
    RADIUS = 2  # temporal smoothing: frames before and after

    @classmethod
    def define_schema(cls):
        return _schema("ComfyProAlpha_EdgeColour", [
            io.Image.Input("images"), io.Int.Input("start", default=0), io.Mask.Input("alpha"),
            io.String.Input("info", default="{}"), io.Combo.Input("mode", options=["quick", "perfect"]),
            io.String.Input("filename_prefix", default="comfy_pro_alpha/shot"),
            io.Combo.Input("format", options=["png16", "exr"]), io.Int.Input("first_frame", default=1),
            io.Boolean.Input("comp", default=True),
            io.Image.Input("clean_plate", optional=True), io.Mask.Input("others", optional=True),
        ], [io.Image.Output("comp")], output_node=True)

    @classmethod
    def execute(cls, images, start, alpha, info, mode, filename_prefix, format, first_frame, comp=True,
                clean_plate=None, others=None):
        A = _h(alpha)
        n, H, W = A.shape
        P = _h(images)[start:start + n]
        if P.shape[1] < H or P.shape[2] < W:
            raise ValueError(f"edge colour: plate {P.shape[2]}x{P.shape[1]} is smaller than the alpha {W}x{H}")
        P = P[:, :H, :W]
        ran, note = "quick", ""
        if mode == "perfect":
            if clean_plate is None:
                note = EDGE_NOTE
            else:
                ran = "perfect"
        s = max(W, H) / 1920  # pixel scale vs 1080p, the same for portrait plates
        out_dir = _out_dir(filename_prefix)
        comp_t = torch.empty((n, H, W, 3), dtype=torch.float32) if comp else torch.zeros(1, 64, 64, 3)
        every = max(1, n // 48)
        size = (min(960, W), int(min(960, W) * H / W))
        small = {}
        raw = {}  # frame -> (unsmoothed F, linear; grey plate for the flow), only the sliding window is kept

        C = _h(clean_plate) if ran == "perfect" else None
        O = _h(others) if others is not None and ran == "perfect" else None

        def frame(f):
            if f not in raw:
                I = E.srgb_to_linear(P[f])
                F = E.quick_colour(I, A[f], s)
                if C is not None:  # perfect: unmix against the clean plate (made on the proxy: scale it up)
                    c = C[f].astype(np.float32) / 255 if C.dtype == np.uint8 else C[f]
                    B = E.srgb_to_linear(cv2.resize(E._f32(c), (W, H), interpolation=cv2.INTER_CUBIC))
                    B = E.match_clean(B, I, A[f], s)
                    o = None if O is None else cv2.resize(E._f32(O[f]), (W, H), interpolation=cv2.INTER_LINEAR)
                    F = E.unmix_colour(I, A[f], B, F, E.clean_confidence(B, I, A[f], o, s))
                raw[f] = (F, E.to_gray(P[f:f + 1])[0])
            return raw[f]

        for f in range(n):
            lo, hi = max(0, f - cls.RADIUS), min(n, f + cls.RADIUS + 1)
            win = list(range(lo, hi))
            F = E.smooth_colour([frame(k)[0] for k in win], [A[k] for k in win],
                                np.stack([frame(k)[1] for k in win]), cls.RADIUS, index=f - lo)
            for k in [k for k in raw if k < f - cls.RADIUS]:
                del raw[k]
            a = E._f32(A[f])
            _save_rgba(F, a, out_dir, format, first_frame + f)
            if comp:
                comp_t[f] = torch.from_numpy(E.linear_to_srgb(F * a[..., None]))
            if f % every == 0:
                small[f] = cv2.resize(F, size, interpolation=cv2.INTER_AREA)
        ui_info = {**json.loads(info or "{}"), "edge_colour": ran, "edge_note": note}
        return io.NodeOutput(comp_t, ui={"pro_alpha": [ui_info], "previews": _previews(P, A, every, clean=small)})


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


NODES = [Plate, LoadVideo, Frame, PlateProxy, TrackMasks, MaskMax, EmptyPrompt, Refine, Isolate, Depth, ChunkPad, AttnChunk,
         DecodeAlpha, Stitch, Proxy, NativeIsolate, TileGuide, DecodeStitch, DecodeStitchRGB, TileMerge, NativeEdges, Final, EdgeColour,
         DetectReport, PartReport]
