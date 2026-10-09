"""Selective Matte: one node, ONYX-style. Pick a subject (person, animal, vehicle, prop) in the plate, get an
LTX-quality alpha for only that subject.

At run time the node expands into SAM3 tracks -> selection / ViTMatte / stabilize -> LTX alpha-gen (chunked)
-> [isolated plate -> LTX again] -> v12 blend -> save. Each stage is cached separately.
"""
import json
import math
import os

import folder_paths
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from . import _engine as E
from .internal import plate_fingerprint

NONE = "(none found)"
BUILTIN = "built-in empty prompt (no text encoder needed)"
# widget -> (models folder, preferred file, where to get it): shown as tooltips and in "missing model" errors
MODELS = {
    "ltx_model": ("diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    "alpha_lora": ("loras", "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"),
    "text_encoder": ("text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    "video_vae": ("vae", "ltx-2.5-video-vae-conv-bf16.safetensors"),
    "audio_vae": ("vae", "ltx-2.5-audio-vae-bf16.safetensors"),
}
QUALITY = ["Best (v12, 2 LTX passes)", "Fast (v7, 1 LTX pass)"]
RENDER = ["full shot", "preview chunk"]
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def _models(widget):
    """Files in the widget's models folder, the recommended one first. Re-read every time the node definitions
    are fetched, so after dropping a model in the folder, pressing R (refresh) in ComfyUI lists it."""
    kind, prefer = MODELS[widget]
    names = folder_paths.get_filename_list(kind)
    best = next((n for n in names if n.replace("\\", "/").split("/")[-1] == prefer), None)
    best = best or next((n for n in names if prefer.rsplit("-", 1)[0] in n), None)
    return ([best] if best else []) + [n for n in names if n != best] or [NONE]


def _model_input(widget, label):
    kind, prefer = MODELS[widget]
    tip = f"Recommended: {prefer} | folder: ComfyUI/models/{kind}/ (press R after adding files)"
    if widget == "text_encoder":
        # Alpha-gen uses an empty prompt; its conditioning ships with the pack, so the encoder is optional.
        files = [n for n in _models(widget) if n != NONE]
        return io.Combo.Input(widget, options=[BUILTIN] + files, default=BUILTIN, display_name=label, tooltip=(
            f"{BUILTIN}: same result, skips loading the 13 GB encoder. Or pick {prefer} from ComfyUI/models/{kind}/"))
    return io.Combo.Input(widget, options=_models(widget), display_name=label, tooltip=tip)


def _kitchen_attention():
    """Core 'Model Attention Backend' node with the Comfy Kitchen int8 kernel (ComfyUI 0.38+, Nvidia/AMD)."""
    import nodes
    import comfy.ldm.modules.attention as attn
    return "ModelAttentionBackend" in nodes.NODE_CLASS_MAPPINGS and getattr(attn, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)


def _vitmatte_options():
    try:
        d = folder_paths.get_folder_paths("vitmatte")
    except KeyError:
        d = []
    found = sorted({n for p in d if os.path.isdir(p) for n in os.listdir(p) if os.path.isdir(os.path.join(p, n))})
    return found + ["auto-download"]


class SelectiveMatte(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SelectiveMatte",
            display_name="Selective Matte",
            category="SelectiveMatte",
            description="Matte ONE chosen subject with LTX-2.5 alpha-gen edge quality. Type what to find (person, dog, "
                        "car...), click Detect, click the subject; alt-click cuts parts, shift-click adds parts.",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("images", tooltip=(
                    "The plate. Connect Load Video -> Get Video Components, Load Plate Sequence, or any crop/resize "
                    "chain. Then click 'Load plate' in the node.")),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001,
                               tooltip="Plate frame rate (connect Get Video Components / Load Plate Sequence fps)"),
                io.Int.Input("first_frame", default=1, min=0, max=10**7,
                             tooltip="Frame number of the first image (1001 for VFX plates): output files use it"),
                io.Combo.Input("quality", options=QUALITY, default=QUALITY[0]),
                io.Combo.Input("render", options=RENDER, default=RENDER[0],
                               tooltip="preview chunk = first 49 frames from the pick frame"),
                io.String.Input("picks", default="{}", multiline=False, tooltip="Set by the picker; JSON."),
                io.String.Input("filename_prefix", default="selective_matte/shot"),
                io.Combo.Input("format", options=["png16", "exr"], default="png16"),
                # models (always visible)
                _model_input("ltx_model", "LTX-2.5 model"),
                _model_input("alpha_lora", "alpha-gen IC-LoRA"),
                _model_input("text_encoder", "text encoder"),
                _model_input("video_vae", "video VAE (conv)"),
                _model_input("audio_vae", "audio VAE"),
                io.Combo.Input("vitmatte", options=_vitmatte_options(), display_name="ViTMatte",
                               tooltip="Folder in ComfyUI/models/vitmatte/, or auto-download on first run"),
                # advanced
                io.Int.Input("chunk", default=49, min=9, max=145, step=8, tooltip="LTX frames per render (8n+1)"),
                io.Int.Input("stride", default=40, min=1, max=144),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1, control_after_generate=False),
                io.Int.Input("erode", default=10, min=0, max=64), io.Int.Input("dilate", default=15, min=1, max=64),
                io.Int.Input("near_px", default=15, min=1, max=64, tooltip="v7 zone around other excluded subjects (Best)"),
                io.Int.Input("stabilize_radius", default=2, min=0, max=6),
                io.Float.Input("sam3_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("fast_attention", default=True, tooltip=(
                    "Comfy Kitchen int8 attention for LTX: about 36% faster sampling, matte difference ~0.004 "
                    "(7x smaller than changing the seed). Turn off to use standard PyTorch attention.")),
            ],
            outputs=[io.Mask.Output("alpha"), io.Image.Output("comp")],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def validate_inputs(cls, picks, **kw):
        missing = [f"{w}: put {prefer} in ComfyUI/models/{kind}/ then press R"
                   for w, (kind, prefer) in MODELS.items() if kw.get(w) == NONE]
        if missing:
            return "Missing model(s): " + "; ".join(missing)
        try:
            p = json.loads(picks or "{}")
        except ValueError:
            return "picks is not valid JSON"
        if not p.get("target"):
            return "No subject picked: click the subject in the node (anything in the plate, or a detection after Detect)."
        return True

    @classmethod
    def execute(cls, images, fps, first_frame, quality, render, picks, filename_prefix, format, ltx_model, alpha_lora, text_encoder,
                video_vae, audio_vae, vitmatte, chunk, stride, seed, erode, dilate, near_px, stabilize_radius,
                sam3_threshold, fast_attention=True):
        p = json.loads(picks)
        n_video, H, W = images.shape[:3]
        # Sub-nodes link straight to whatever feeds our `images` input (no copy of the batch).
        src = cls.hidden.dynprompt.get_node(cls.hidden.unique_id)["inputs"]["images"]
        if p.get("plate") and p["plate"] != plate_fingerprint(images):
            print("[SelectiveMatte] warning: the plate changed since the subject was picked; picks may be off")
        start = int(p["target"]["frame"])
        count = n_video - start
        L = min(chunk, E.ltx_frames(count) if count >= chunk else 8 * math.ceil((count - 1) / 8) + 1)
        if render == RENDER[1]:
            count = min(count, L)
        starts = E.chunk_starts(count, L, stride)
        if len(starts) > 24:
            raise ValueError(f"{count} frames needs {len(starts)} chunks (max 24): raise stride or split the shot")
        best = quality == QUALITY[0]
        prefix = filename_prefix + ("_preview" if render == RENDER[1] else "")

        g = GraphBuilder()
        vid = g.node("SelectiveMatte_LoadVideo", images=src, fps=fps)
        sam = g.node("LoadSAM3Model", precision="auto")

        last = start + count - 1

        def track(mode, frame, **prompt):
            """One SAM3 object, tracked forward from its prompt frame to the end of the processed range."""
            seg = g.node("SAM3VideoSegmentation", video=vid.out(0), prompt_mode=mode, frame_idx=int(frame),
                         score_threshold=sam3_threshold, **prompt)
            prop = g.node("SAM3Propagate", sam3_model_config=sam.out(0), video_state=seg.out(0),
                          start_frame=int(frame), end_frame=last, direction="forward")
            return g.node("SelectiveMatte_TrackMasks", masks=prop.out(0), start=start, count=count,
                          width=W, height=H).out(0)

        def box_track(b):
            cx, cy, w, h = b["box"]
            box = g.node("SAM3CreateBox", center_x=cx, center_y=cy, width=w, height=h, is_positive=True)
            return track("box", b["frame"], positive_boxes=g.node("SAM3CombineBoxes", box_1=box.out(0)).out(0))

        def point_track(pt):
            neg = pt.get("neg", [])
            pts = g.node("SAM3CombinePoints", **{f"point_{i + 1}": g.node("SAM3CreatePoint", x=x, y=y, is_foreground=True).out(0)
                                                for i, (x, y) in enumerate(pt.get("points") or [[pt["x"], pt["y"]]])})
            kw = {"positive_points": pts.out(0)}
            if neg:
                kw["negative_points"] = g.node("SAM3CombinePoints", **{
                    f"point_{i + 1}": g.node("SAM3CreatePoint", x=x, y=y, is_foreground=False).out(0)
                    for i, (x, y) in enumerate(neg)}).out(0)
            return track("point", pt["frame"], **kw)

        def union(masks):
            out = None
            for m in masks:
                out = m if out is None else g.node("SelectiveMatte_MaskMax", a=out, b=m).out(0)
            return out

        # Target from a detection (box) or from a plain click anywhere (SAM3 point prompt, no Detect needed).
        target = box_track(p["target"]) if "box" in p["target"] else point_track(p["target"])
        in_range = lambda items: [x for x in items if int(x["frame"]) <= last]  # parts first seen after a preview
        refine_kw = {k: v for k, v in (("adds", union([point_track(a) for a in in_range(p.get("adds", []))])),
                                       ("cuts", union([point_track(c) for c in in_range(p.get("cuts", []))])),
                                       ("others", union([box_track(o) for o in in_range(p.get("others", []))]))) if v is not None}
        plate = g.node("SelectiveMatte_Plate", images=src, start=start, count=count)
        ref = g.node("SelectiveMatte_Refine", plate=plate.out(0), target=target, vitmatte=vitmatte,
                     erode=erode, dilate=dilate, radius=stabilize_radius, **refine_kw)
        sel, refined, others = ref.out(0), ref.out(1), ref.out(2)

        # LTX-2.5 alpha-gen stack, settings from kids_ltx_alphagen_1920_49f_API.json
        unet = g.node("UNETLoader", unet_name=ltx_model, weight_dtype="default")
        lora = g.node("LTXICLoRALoaderModelOnly", model=unet.out(0), lora_name=alpha_lora, strength_model=1.0)
        model = lora.out(0)
        if fast_attention:
            if _kitchen_attention():
                model = g.node("ModelAttentionBackend", model=model, attention="comfy kitchen attention").out(0)
            else:
                print("[SelectiveMatte] Comfy Kitchen attention unavailable (update ComfyUI); using PyTorch attention")
        if text_encoder == BUILTIN:
            empty = g.node("SelectiveMatte_EmptyPrompt").out(0)
            cond = g.node("LTXVConditioning", positive=empty, negative=empty, frame_rate=fps)
        else:
            clip = g.node("CLIPLoader", clip_name=text_encoder, type="ltxv")
            cond = g.node("LTXVConditioning", positive=g.node("CLIPTextEncode", clip=clip.out(0), text="").out(0),
                          negative=g.node("CLIPTextEncode", clip=clip.out(0), text="").out(0), frame_rate=fps)
        vae = g.node("VAELoader", vae_name=video_vae)
        avae = g.node("VAELoader", vae_name=audio_vae)
        latent = g.node("EmptyLTXVLatentVideo", width=W + (-W) % 32, height=H + (-H) % 32, length=L, batch_size=1)
        alat = g.node("LTXVEmptyLatentAudio", frames_number=L, frame_rate=int(round(fps)), batch_size=1, audio_vae=avae.out(0))
        noise = g.node("RandomNoise", noise_seed=seed)
        sampler = g.node("KSamplerSelect", sampler_name="euler_ancestral")
        sigmas = g.node("ManualSigmas", sigmas=SIGMAS)
        plan = json.dumps({"n": count, "h": H, "w": W, "chunk": L, "starts": starts})

        def ltx_pass(images):
            chunks = {}
            for i, s in enumerate(starts):
                img = g.node("SelectiveMatte_ChunkPad", images=images, start=s, length=L)
                guide = g.node("LTXAddVideoICLoRAGuide", positive=cond.out(0), negative=cond.out(1), vae=vae.out(0),
                               latent=latent.out(0), image=img.out(0), frame_idx=0, strength=1.0,
                               latent_downscale_factor=lora.out(1), crop="disabled", use_tiled_encode=False,
                               tile_size=256, tile_overlap=64)
                av = g.node("LTXVConcatAVLatent", video_latent=guide.out(2), audio_latent=alat.out(0))
                guider = g.node("CFGGuider", model=model, positive=guide.out(0), negative=guide.out(1), cfg=1.0)
                samp = g.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0), sampler=sampler.out(0),
                              sigmas=sigmas.out(0), latent_image=av.out(0))
                sep = g.node("LTXVSeparateAVLatent", av_latent=samp.out(0))
                crop = g.node("LTXVCropGuides", positive=guide.out(0), negative=guide.out(1), latent=sep.out(0))
                dec = g.node("VAEDecodeTiled", samples=crop.out(2), vae=vae.out(0), tile_size=512, overlap=64,
                             temporal_size=64, temporal_overlap=8)
                chunks[f"chunk_{i}"] = dec.out(0)
            return g.node("SelectiveMatte_Stitch", plan=plan, **chunks).out(0)

        ltx = ltx_pass(plate.out(0))
        final_kw = {}
        if best:
            iso = g.node("SelectiveMatte_Isolate", plate=plate.out(0), selection=sel, others=others, grow=6, blur=6.0)
            final_kw["ltx_isolated"] = ltx_pass(iso.out(0))
        final = g.node("SelectiveMatte_Final", plate=plate.out(0), ltx=ltx, selection=sel, refined=refined,
                       others=others, near_px=near_px, filename_prefix=prefix, format=format,
                       first_frame=first_frame + start, save=True, qc=True, **final_kw)
        return io.NodeOutput(final.out(0), final.out(1), expand=g.finalize())
