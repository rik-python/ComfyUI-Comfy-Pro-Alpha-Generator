"""Node-level tests for the v14 stages (TileMerge, NativeEdges, TileGuide, Final's native merge).

They import the pack the way ComfyUI does, so they need ComfyUI's core on sys.path: set COMFYUI_DIR, or they look in
the default Comfy Desktop location and are skipped when it is not there.
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_DIR", r"C:\Users\rikin\ComfyUI-Installs\ComfyUI\ComfyUI")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(COMFY, "comfy")), reason="ComfyUI core not found")


@pytest.fixture(scope="module")
def pack():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    if "cpa_pack" not in sys.modules:
        spec = importlib.util.spec_from_file_location("cpa_pack", os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cpa_pack"] = mod
        spec.loader.exec_module(mod)
    return types.SimpleNamespace(I=sys.modules["cpa_pack.internal"])


def _link(n):
    return [str(n), 0]


STACK = {k: _link(i) for i, k in enumerate(["pos", "neg", "vae", "lora_scale", "alat", "model", "noise", "sampler",
                                            "sigmas"], start=10)}


def _config(W, H, count=1, starts=(0,)):
    return json.dumps({"W": W, "H": H, "start": 0, "count": count, "L": 49, "starts": list(starts),
                       "links": {"selection": _link(1), "others": _link(2), "plate": _link(3), "stack": STACK}})


# ------------------------------------------------------------------ TileMerge / _native_alpha

def test_tile_merge_coverage_is_one_frame(pack):
    # coverage only depends on which tiles were rendered (chosen once for the whole range): one frame, not n
    W, H = 64, 32
    plan = json.dumps({"n": 3, "W": W, "H": H, "grid": [[0, 0, W, H]], "used": [0]})
    res = pack.I.TileMerge.execute(plan=plan, tile_0=torch.full((3, H, W), 0.5, dtype=torch.float16)).result
    assert tuple(res[0].shape) == (3, H, W)
    assert tuple(res[1].shape) == (1, H, W)


def test_native_alpha_is_float32_and_takes_one_frame_coverage(pack):
    alpha = np.full((2, 16, 32), 0.5, np.float32)
    others = np.zeros((2, 16, 32), np.float32)
    out = pack.I._native_alpha(alpha, (32, 64), torch.full((2, 32, 64), 0.5), torch.ones((1, 32, 64)), others)
    assert out.dtype == np.float32  # same as the 1080p path (GrowMask and friends reject float16)
    assert out.shape == (2, 32, 64)


def test_native_alpha_without_tiles_is_scaled_proxy(pack):
    alpha = np.random.default_rng(0).random((2, 16, 32)).astype(np.float32)
    others = np.zeros((2, 16, 32), np.float32)
    plain = pack.I._native_alpha(alpha, (32, 64), None, None, others)
    empty = pack.I._native_alpha(alpha, (32, 64), torch.zeros((0, 1, 1)), torch.zeros((0, 1, 1)), others)
    assert np.array_equal(plain, empty)


# ------------------------------------------------------------------ NativeEdges

def test_native_edges_schema_keys_cache_on_everything_the_tiles_use(pack):
    names = {i.id for i in pack.I.NativeEdges.define_schema().inputs}
    assert {"plate", "others", "model", "positive", "negative", "vae", "audio_latent", "lora_scale", "noise",
            "sampler", "sigmas"} <= names


def test_native_edges_empty_selection_allocates_nothing(pack):
    sel = torch.zeros((4, 1080, 1920), dtype=torch.float16)
    res = pack.I.NativeEdges.execute(selection=sel, config=_config(3840, 2160, count=4))
    tiles, coverage, count = res.result
    assert count == 0
    assert tiles.shape[0] == 0 and coverage.shape[0] == 0


def test_native_edges_too_many_tiles_raises(pack):
    yy, xx = np.indices((1080, 1920))
    sel = torch.from_numpy(((yy // 24 + xx // 24) % 2).astype(np.float16))[None]  # edges everywhere
    with pytest.raises(ValueError, match="tiles"):
        pack.I.NativeEdges.execute(selection=sel, config=_config(12000, 6750))


def test_native_edges_expands_one_pass_per_tile_and_chunk(pack):
    sel = torch.zeros((1, 1080, 1920), dtype=torch.float16)
    sel[:, :, :960] = 1  # left half: the edge runs down the middle column of a UHD plate
    res = pack.I.NativeEdges.execute(selection=sel, config=_config(3840, 2160, count=60, starts=(0, 40)))
    n_tiles = res.result[2]
    graph = res.expand
    kinds = [n["class_type"] for n in graph.values()]
    assert n_tiles > 0
    assert kinds.count("ComfyProAlpha_NativeIsolate") == 1
    assert kinds.count("ComfyProAlpha_TileGuide") == 2 * n_tiles  # one per tile per LTX chunk
    assert kinds.count("ComfyProAlpha_DecodeStitch") == n_tiles
    assert kinds.count("ComfyProAlpha_TileMerge") == 1
    guiders = [n for n in graph.values() if n["class_type"] == "CFGGuider"]
    assert all(g["inputs"]["model"] == STACK["model"] for g in guiders)


# ------------------------------------------------------------------ TileGuide

def test_tile_guide_unwraps_v1_and_v3_outputs_and_names_missing_node(pack, monkeypatch):
    import nodes

    class V3:
        @classmethod
        def execute(cls, **kw):
            return types.SimpleNamespace(result=("p", "n", "l"))

    class V1:
        FUNCTION = "go"

        def go(self, **kw):
            return ("p1", "n1", "l1")

    args = dict(images=torch.zeros((2, 64, 64, 3)), x=0, y=0, width=64, height=64, start=0, length=2, end=-1,
                positive=[], negative=[], vae=None, latent={}, latent_downscale_factor=1.0)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LTXAddVideoICLoRAGuide", V3)
    assert pack.I.TileGuide.execute(**args).result == ("p", "n", "l")
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LTXAddVideoICLoRAGuide", V1)
    assert pack.I.TileGuide.execute(**args).result == ("p1", "n1", "l1")
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, "LTXAddVideoICLoRAGuide")
    with pytest.raises(RuntimeError, match="LTXAddVideoICLoRAGuide"):
        pack.I.TileGuide.execute(**args)
