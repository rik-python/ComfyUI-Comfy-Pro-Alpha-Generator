"""Load Plate Sequence: a VFX image sequence (PNG/JPG/TIF/EXR) as IMAGE + fps + its own first frame number."""
import os

import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io

from . import _video as V


class LoadPlateSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SelectiveMatte_LoadPlateSequence",
            display_name="Load Plate Sequence",
            category="SelectiveMatte",
            description="Image sequence -> IMAGE. Keeps the plate's frame numbers (first_frame) so the matte comes "
                        "out as alpha_01001... EXR is treated as scene-linear and converted to sRGB.",
            inputs=[
                io.String.Input("path", default="", tooltip=(
                    "Folder of frames, or a pattern: D:/plates/sh010/plate.####.exr (or %04d). Absolute, or relative "
                    "to ComfyUI/input.")),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001),
                io.Int.Input("first", default=0, min=0, tooltip="First frame NUMBER to load (0 = from the start)"),
                io.Int.Input("last", default=0, min=0, tooltip="Last frame NUMBER to load (0 = to the end)"),
            ],
            outputs=[io.Image.Output("images"), io.Float.Output("fps"), io.Int.Output("first_frame")],
        )

    @classmethod
    def validate_inputs(cls, path, **kw):
        try:
            V.seq_files(path)
        except (FileNotFoundError, OSError) as e:
            return str(e)
        return True

    @classmethod
    def fingerprint_inputs(cls, path, fps, first, last):
        files = V.seq_files(path)
        return (path, fps, first, last, len(files), os.path.getmtime(files[0][1]), os.path.getmtime(files[-1][1]))

    @classmethod
    def execute(cls, path, fps, first, last):
        files = [(n, p) for n, p in V.seq_files(path)
                 if n is None or ((not first or n >= first) and (not last or n <= last))]
        if not files:
            raise ValueError(f"no frames in range {first}-{last} for {path}")
        pbar = comfy.utils.ProgressBar(len(files))
        frames = []
        for _, p in files:
            frames.append(V.read_image(p))
            pbar.update(1)
        images = torch.from_numpy(np.stack(frames)).float() / 255.0
        first_frame = files[0][0] if files[0][0] is not None else 1
        return io.NodeOutput(images, fps, first_frame)
