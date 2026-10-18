"""Comfy Pro Alpha Generator by Rikin Patel (Comfy Compositing): SAM3 picks the subject, LTX-2.5 alpha-gen draws the edges (V3 API)."""
import os

try:
    import folder_paths
except ImportError:  # imported outside ComfyUI (pytest sets up the pack as a package for tests/): register nothing
    folder_paths = None

WEB_DIRECTORY = "./web"
__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]

if folder_paths is not None:
    from typing_extensions import override

    from comfy_api.latest import ComfyExtension, io

    # models/vitmatte/<hf-folder> (e.g. vitmatte-base-composition-1k); auto-downloaded on first run if empty.
    folder_paths.add_model_folder_path("vitmatte", os.path.join(folder_paths.models_dir, "vitmatte"))

    from .internal import NODES as INTERNAL  # noqa: E402
    from .node import ComfyProAlphaGenerator, LegacySelectiveMatte  # noqa: E402
    from .sequence import LegacyLoadPlateSequence, LoadPlateSequence  # noqa: E402

    class ComfyProAlphaExtension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [ComfyProAlphaGenerator, LoadPlateSequence, LegacySelectiveMatte, LegacyLoadPlateSequence, *INTERNAL]

    async def comfy_entrypoint() -> ComfyProAlphaExtension:
        return ComfyProAlphaExtension()
