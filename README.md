# ComfyUI-SelectiveMatte

One node that mattes **one chosen person** in a multi-person shot, with **LTX-2.5 alpha-gen** edge and hair quality.
LTX alpha-gen mattes everyone and can't be told who. This node lets SAM3 pick the subject, then lets LTX draw the edges.

## Using it

1. Add the **Selective Matte** node: double-click the canvas and type `selective`.
2. Pick or upload the plate in `video`.
3. Scrub to a frame where the person is fully visible, then click **Detect people**.
4. **Click the person** to matte. They turn orange. Everyone else is excluded automatically.
5. Fix parts if needed, on any frame where the part is visible:

   | Action | What it does |
   |---|---|
   | **Alt-click** | **Cut** a part: someone else's hand on the subject |
   | **Shift-click** | **Add** a part: the subject's own hand that SAM3 missed |
   | **Ctrl-click** | Remove a cut or add marker |

   Place a cut or add on the **first frame** that part appears. Tracking only runs forward.
   Each click gets automatic "not this" points, so a single click gives just the part:
   - **Add:** the nearby other people are marked "not this".
   - **Cut:** the subject is marked "not this".

   The part shows as a red or green overlay. If a click still grabs a whole person, a red warning tells you to Ctrl-click it away and click closer to the hand.
6. Click **Preview chunk** (first 49 frames). Check the **Result** tab: drag to wipe plate against comp, and press `comp / alpha` to switch views.
7. Click **Run full shot**.

The alpha is saved to `ComfyUI/output/<filename_prefix>/alpha_00001.png …`. Frame numbers match the plate, starting at 1.
It is 16-bit PNG by default; set `format` to `exr` for half-float EXR.

**Quality**

| Preset | What it does |
|---|---|
| **Best (v12)** | 2 LTX passes. Pass 2 runs on a plate with everyone else painted out, so LTX draws the subject's edges itself. |
| **Fast (v7)** | 1 LTX pass. About half the time; edges against other people are a little weaker. |

**QC badge** (shown after each run)

| Badge | Meaning | Green when |
|---|---|---|
| flicker | Motion-compensated flicker, compared with LTX alone | ≤ 1.3x |
| hair kept | Share of LTX's soft hair edge that survived | ≥ 95% |

`⚙ Advanced` holds the model files, the chunk/stride/seed settings, the trimap erode/dilate, the near-people zone and the SAM3 threshold.

## Install

**1. Clone into `ComfyUI/custom_nodes`:**
```
git clone https://github.com/rik-python/ComfyUI-SelectiveMatte.git
```

**2. Required node packs** (ComfyUI Manager can install both):
- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)

**3. Models.** The Lightricks repos are gated: run `hf auth login` once.

| File | Folder |
|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models` |
| `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` ([Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen)) | `models/loras` |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders` |
| `ltx-2.5-video-vae-conv-bf16.safetensors` (**conv** VAE; the diffusion VAE can't encode guides) | `models/vae` |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae` |
| ViTMatte `hustvl/vitmatte-base-composition-1k` (auto-downloads on the first run) | `models/vitmatte/` |

**4. Hardware.** You need a 32 GB GPU (RTX 5090 class) and 64 GB+ of system RAM.
It renders 1920x1088 x 49 frames per LTX chunk; longer shots are split into overlapping chunks and cross-faded.

## How it works

On **Queue**, the node expands into a normal ComfyUI graph:

1. **SAM3 tracks** for the target, each excluded person, and each cut and add.
2. Build the selection, then refine it with ViTMatte.
3. **Stabilize** the selection and ViTMatte edges over time with optical flow.
4. **LTX pass 1** on the plate, chunked and cross-faded.
5. **Isolated plate**: everyone else inpainted out.
6. **LTX pass 2** on the isolated plate.
7. **v12 blend**: LTX edges everywhere, with ViTMatte within about 15 px of other people (where LTX has no edge information).
8. Save the alpha and write the QC scores.

Each stage is its own cached node. Changing the quality preset, or running the full shot after a preview, reuses the stages that already finished.

## Limits

- Tracks start at the frame you detected on. Frames before it come out empty.
- Where two people overlap, the edge position between them comes from SAM3/ViTMatte, not LTX.
- Light fringes on hair are colour spill from the plate. Despill in comp.
- Shots are limited to 24 chunks (about 960 frames at stride 40). Split longer shots.
