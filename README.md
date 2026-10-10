# ComfyUI-SelectiveMatte

One node that mattes **one chosen subject** (a person, animal, vehicle or prop) in a busy shot, with **LTX-2.5 alpha-gen**
edge and hair quality. LTX alpha-gen mattes every main subject and can't be told which one. This node lets SAM3 pick the
subject from a word you type, then lets LTX draw the edges.

## Using it

1. Open `example_workflow.json` (video) or `example_sequence_workflow.json` (image sequence). Or wire it yourself:

   | Plate | Wiring |
   |---|---|
   | **Video** | Load Video → Get Video Components → `images` + `fps` |
   | **Image sequence** | **Load Plate Sequence** → `images` + `fps` + `first_frame` |

   **Load Plate Sequence** reads a folder (`D:/plates/sh010`) or a pattern (`D:/plates/sh010/plate.####.exr` or `plate.%04d.exr`).
   - Formats: PNG, JPG, TIF, EXR. EXR is treated as scene-linear and converted to sRGB.
   - `first` and `last` set the frame range.
   - Output frames keep the plate's numbers: `plate.1001` gives `alpha_01001`.

   Any nodes can sit in between (crop, resize, colour). The node mattes whatever arrives at `images`.
   **Best quality at 1920x1088** (a 1920x1080 plate is padded automatically): LTX alpha-gen is tuned for that size.
   Other sizes work but edges can be softer. For 4K plates, matte a 1080p proxy and scale the alpha up.
   The picker shows the plate size as a chip (amber when it is not 1920x1080/1088).
2. Click **Load plate** in the node. It runs the nodes feeding `images` and fills the scrubber.
   Click it again whenever you change the crop or the clip. If the plate changes after you picked someone, the node warns you to re-detect.
3. Scrub to a frame where the subject is fully visible. Type what to find in the box next to **Detect** and press **Detect** (or Enter):

   | Type | Finds |
   |---|---|
   | `person` (default) | every person |
   | `dog`, `red car`, `bike`, `woman` | any SAM3 concept |
   | `dog, person` | both. Use this to also paint out the people near a dog |

   Each detection is outlined and labelled (`2 dog`).
4. **Click the subject** to matte. It turns orange. Other detections are excluded automatically.

   **No typing needed:** click anything that is not a detection (a table, a lamp, a prop), with or without Detect.
   SAM3 segments what you clicked and it becomes the subject. Then **Shift-click** more objects to add them:
   the man, then his glass, then the table gives one matte of all three.
5. Fix parts if needed, on any frame where the part is visible:

   | Action | What it does |
   |---|---|
   | **Alt-click** | **Cut** a part: someone else's hand on the subject |
   | **Shift-click** | **Add** a part: the subject's own hand that SAM3 missed |
   | **Ctrl-click** | Remove a cut or add marker |

   Place a cut or add on the **first frame** that part appears. Tracking only runs forward.
   Each click gets automatic "not this" points, so a single click gives just the part:
   - **Add:** the other detections nearby are marked "not this".
   - **Cut:** the subject is marked "not this".

   The part shows as a red or green overlay. If a click still grabs a whole subject, a red warning tells you to Ctrl-click it away and click closer to the part.
6. Click **Preview chunk** (first 49 frames). Check the **Result** tab: drag to wipe plate against comp, and press `comp / alpha` to switch views.
7. Click **Run full shot**.

The alpha is saved to `ComfyUI/output/<filename_prefix>/alpha_00001.png …`. Frame numbers match the plate, starting at 1.
It is 16-bit PNG by default; set `format` to `exr` for half-float EXR.

**Quality**

| Preset | What it does |
|---|---|
| **Best (v12)** | 2 LTX passes. Pass 2 runs on a plate with the other detections painted out, so LTX draws the subject's edges itself. |
| **Fast (v7)** | 1 LTX pass. About half the time; edges against other subjects are a little weaker. |

**QC badge** (shown after each run)

| Badge | Meaning | Green when |
|---|---|---|
| flicker | Motion-compensated flicker, compared with LTX alone | ≤ 1.3x |
| soft edge kept | Share of LTX's soft edge (hair, fur, dust) that survived | ≥ 95% |
| LTX cover | How much of the subject LTX matted. Where it is low, the node uses the ViTMatte edge instead | ≥ 90% |

**Models** are always visible on the node: LTX model, alpha-gen IC-LoRA, text encoder, video VAE, audio VAE, ViTMatte. Hover a model selector to see the recommended file and its folder.
After you copy model files into the folders, press **R** in ComfyUI to refresh the lists. A missing model is reported by name when you queue.

`⚙ Advanced` holds frames per chunk, overlap frames (chunk joins are cross-faded over these), the seed, temporal smoothing (frames before/after each frame used to steady the SAM3/ViTMatte edges, default 2), the trimap erode/dilate, the zone near other subjects, the SAM3 threshold and `fast_attention`.

**Speed.** `fast_attention` (on by default) runs LTX with Comfy Kitchen int8 attention, built into ComfyUI 0.38+.
LTX sampling is about 36% faster. The matte changes by about 0.004 on edges, 7x less than changing the seed.
Flicker and hair stay the same. On older ComfyUI the node falls back to standard attention.

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
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (**optional**: the pack ships the empty-prompt conditioning, so `text encoder` defaults to built-in) | `models/text_encoders` |
| `ltx-2.5-video-vae-conv-bf16.safetensors` (**conv** VAE; the diffusion VAE can't encode guides) | `models/vae` |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae` |
| ViTMatte `hustvl/vitmatte-base-composition-1k` (auto-downloads on the first run) | `models/vitmatte/` |

**4. Hardware.** Any Nvidia GPU with **12 GB+ VRAM**, and **16 GB+ of system RAM** with the models on an NVMe SSD.
It renders 1920x1088 x 49 frames per LTX chunk; longer shots are split into overlapping chunks and cross-faded.
See [Low VRAM / low RAM](#low-vram--low-ram).

## Low VRAM / low RAM

ComfyUI 0.38+ streams model weights to the GPU as it needs them (dynamic VRAM), so the 22B LTX model runs on small
GPUs. The catch: the weights then wait in **system RAM** (about 24 GB) unless ComfyUI can also stream them from disk.

**Disk streaming is the low-RAM switch.** ComfyUI turns it on by itself when the model file sits on a fast NVMe SSD.
On other drives, add `--fast-disk` to ComfyUI's launch arguments. To check: the ComfyUI log prints
`Model storage policy: fast_disk=True` with the LTX model's path when it is on.

Measured on the dogs preview (49 frames, Best, int8 model, RTX 5090; times are from a fresh start, model loading included):

| Setup | Peak RAM | Time | Matte |
|---|---|---|---|
| Model in RAM (no disk streaming) | 40 GB | | reference |
| Disk streaming | 21 GB | 272 s | identical |
| Disk streaming, machine squeezed to **16 GB RAM** | fits | 333 s | identical |
| 12 GB of VRAM free (simulated) | | about 5% slower | identical |

A real 12 GB card is slower than these numbers because it has less compute, not because it runs out of memory.

**Longer shots** need more working memory (the plate and mattes in RAM). Only the 49-frame preview was tested at
16 GB; on a 16 GB machine, split long shots if a full run runs out of memory.

**GGUF models (optional).** With the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) pack installed, `.gguf`
files show up in the `LTX-2.5 model` list, for example `LTX-2.5-Distilled-Q4_K_M.gguf` (15 GB,
[realrebelai/LTX-2.5_GGUFs](https://huggingface.co/realrebelai/LTX-2.5_GGUFs)). Use it only to save disk space:
GGUF can't be streamed from disk, so it needs **more** RAM than int8 with disk streaming (27 GB vs 21 GB), runs about
30% slower, and its edges differ slightly (edge difference 0.05; they look the same).

## How it works

On **Queue**, the node expands into a normal ComfyUI graph:

1. **SAM3 tracks** for the target, each excluded person, and each cut and add.
2. Build the selection, then refine it with ViTMatte.
3. **Stabilize** the selection and ViTMatte edges over time with optical flow.
4. **LTX pass 1** on the plate, chunked and cross-faded.
5. **Isolated plate**: the other detections inpainted out.
6. **LTX pass 2** on the isolated plate.
7. **v12 blend**: LTX edges everywhere, with ViTMatte within about 15 px of other subjects (where LTX has no edge information).
   Parts of the selection LTX did not matte (low LTX cover, e.g. the table under a person's hands) use the
   ViTMatte matte; the parts LTX did matte keep LTX edges.
8. Save the alpha and write the QC scores.

Each stage is its own cached node. Changing the quality preset, or running the full shot after a preview, reuses the stages that already finished.

## Limits

- Tracks start at the frame you detected on. Frames before it come out empty.
- Where two subjects overlap, the edge position between them comes from SAM3/ViTMatte, not LTX.
- LTX alpha-gen mattes what it reads as the **main** subject. It mattes cars, dust and people well, but a background
  object (a parked car behind the action) can come out empty. Then the LTX cover badge goes red and the node uses
  the ViTMatte edge (ONYX-style quality, no LTX hair detail). Flicker and soft edge then show n/a (they compare against LTX).
- Up to 8 other detections (nearest first) are excluded. Each costs a SAM3 track, about 14 s on a preview.
- If something fully covers the subject (a car driving in front of it), SAM3 can lose the track for the rest of
  the shot. Pick on a frame after the occlusion, or split the shot.
- Light fringes on hair are colour spill from the plate. Despill in comp.
- Shots are limited to 24 chunks (about 960 frames with the default 9 overlap frames). Split longer shots.
