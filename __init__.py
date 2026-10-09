"""ComfyUI-SelectiveMatte - one-node selective matte: SAM3 picks the person, LTX-2.5 alpha-gen draws the edges (V3 API)."""
import os

from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io

# models/vitmatte/<hf-folder> (e.g. vitmatte-base-composition-1k); auto-downloaded on first run if empty.
folder_paths.add_model_folder_path("vitmatte", os.path.join(folder_paths.models_dir, "vitmatte"))

from .internal import NODES as INTERNAL  # noqa: E402
from .node import SelectiveMatte  # noqa: E402
from .sequence import LoadPlateSequence  # noqa: E402

WEB_DIRECTORY = "./web"
__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]


class SelectiveMatteExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SelectiveMatte, LoadPlateSequence, *INTERNAL]


async def comfy_entrypoint() -> SelectiveMatteExtension:
    return SelectiveMatteExtension()
