"""Server routes for the picker UI: video info and single frames as JPEG (for scrubbing)."""
import asyncio
import io as _io
import os

import av
from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer

from ._video import probe

routes = PromptServer.instance.routes
_proxies = {}  # (path, mtime, width) -> list of JPEG bytes, one per frame


def _path(request):
    name = request.query.get("video", "")
    path = folder_paths.get_annotated_filepath(name)
    if not name or not os.path.isfile(path):
        raise web.HTTPNotFound(text=f"video not found: {name}")
    return path


def _build_proxy(path, width):
    """Decode the clip once into small JPEGs so scrubbing is instant."""
    out = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            img = fr.to_image()
            if img.width > width:
                img = img.resize((width, round(img.height * width / img.width)))
            buf = _io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            out.append(buf.getvalue())
    return out


async def _proxy(path, width):
    key = (path, os.path.getmtime(path), width)
    if key not in _proxies:
        if len(_proxies) > 4:
            _proxies.pop(next(iter(_proxies)))
        _proxies[key] = asyncio.get_running_loop().run_in_executor(None, _build_proxy, path, width)
    return await asyncio.shield(_proxies[key]) if asyncio.isfuture(_proxies[key]) else _proxies[key]


@routes.get("/selective_matte/info")
async def info(request):
    return web.json_response(probe(_path(request)))


@routes.get("/selective_matte/frame")
async def frame(request):
    frames = await _proxy(_path(request), int(request.query.get("w", 960)))
    index = max(0, min(int(request.query.get("index", 0)), len(frames) - 1))
    return web.Response(body=frames[index], content_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
