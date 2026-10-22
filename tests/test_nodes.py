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


# ------------------------------------------------------------------ v15 Final info + EdgeColour

@pytest.fixture
def out_dirs(monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "out"))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    return tmp_path


def _final_args(n=2, h=32, w=48):
    m = torch.zeros((n, h, w))
    m[:, :, w // 2:] = 1
    return dict(plate=torch.full((n, h, w, 3), 0.5), ltx=m, selection=m, refined=m, others=torch.zeros((n, h, w)),
                near_px=15, filename_prefix="t/shot", format="png16", first_frame=1001, save=False, qc=False,
                comp=False)


def _edge_args(n=2, h=32, w=48, fmt="exr", comp=False, start=0, images=None):
    a = torch.from_numpy(np.clip((np.arange(w, dtype=np.float32) - 16) / 16, 0, 1))[None, None].repeat(n, h, 1)
    if images is None:
        images = torch.zeros((start + n, h, w, 3))
        images[..., 0], images[..., 1] = 0.8, 0.3
    return dict(images=images, start=start, alpha=a, info=json.dumps({"frames": n}), mode="quick",
                filename_prefix="t/shot", format=fmt, first_frame=1001, comp=comp)


def test_final_returns_info_json(pack, out_dirs):
    res = pack.I.Final.execute(**_final_args())
    assert len(res.result) == 3
    assert json.loads(res.result[2])["frames"] == 2


def test_edge_colour_quick_writes_rgba_files(pack, out_dirs):
    import cv2
    E = sys.modules["cpa_pack._engine"]
    args = _edge_args(fmt="exr")
    pack.I.EdgeColour.execute(**args)
    d = out_dirs / "out" / "t" / "shot"
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    for fr in (1001, 1002):
        img = cv2.imread(str(d / f"rgba_{fr:05d}.exr"), cv2.IMREAD_UNCHANGED)
        assert img.shape == (32, 48, 4)
        a = args["alpha"][0].numpy()
        assert np.abs(img[..., 3] - a).max() < 1e-3
        core = a >= 0.999
        want = E.srgb_to_linear(args["images"][0].numpy())[core] * 1.0
        assert np.abs(img[..., 2::-1][core] - want).max() < 1e-3


def test_edge_colour_png_is_straight_alpha(pack, out_dirs):
    import cv2
    args = _edge_args(fmt="png16")
    pack.I.EdgeColour.execute(**args)
    img = cv2.imread(str(out_dirs / "out" / "t" / "shot" / "rgba_01001.png"), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16 and img.shape[-1] == 4
    core = args["alpha"][0].numpy() >= 0.999
    assert np.abs(img[..., 2::-1][core] / 65535 - args["images"][0].numpy()[core]).max() < 1e-3


def test_edge_colour_comp_is_premultiplied_over_black(pack, out_dirs):
    args = _edge_args(fmt="png16", comp=True)
    comp = pack.I.EdgeColour.execute(**args).result[0].numpy()
    a = args["alpha"].numpy()
    assert comp.shape == (2, 32, 48, 3)
    assert np.abs(comp[a == 0]).max() == 0
    assert np.abs(comp[a >= 0.999] - args["images"].numpy()[a >= 0.999]).max() < 1e-3


def test_edge_colour_reads_plate_from_start(pack, out_dirs):
    images = torch.stack([torch.full((32, 48, 3), v) for v in (0.1, 0.2, 0.3, 0.6, 0.7)])
    args = _edge_args(fmt="png16", comp=True, start=3, images=images)
    comp = pack.I.EdgeColour.execute(**args).result[0].numpy()
    core = args["alpha"][0].numpy() >= 0.999
    assert np.abs(comp[0][core] - 0.6).max() < 1e-3
    assert np.abs(comp[1][core] - 0.7).max() < 1e-3


def test_edge_colour_ui_has_mode_and_previews(pack, out_dirs):
    res = pack.I.EdgeColour.execute(**_edge_args(fmt="png16"))
    info = res.ui["pro_alpha"][0]
    assert info["edge_colour"] == "quick" and info["frames"] == 2
    assert res.ui["previews"]


# ------------------------------------------------------------------ v15 graph wiring

class _Prompt:
    """Minimal dynprompt: our node is "5", its images come from ["1", 0]; optional consumer of our comp output."""

    def __init__(self, comp_wired=False):
        self.nodes = {"5": {"inputs": {"images": ["1", 0]}}}
        if comp_wired:
            self.nodes["9"] = {"inputs": {"image": ["5", 1]}}

    def all_node_ids(self):
        return list(self.nodes)

    def get_node(self, nid):
        return self.nodes[nid]


def _gen_graph(monkeypatch, edge_colour=None, comp_wired=False, size=(1920, 1080), frame=0, **kw):
    gen = sys.modules["cpa_pack.node"].ComfyProAlphaGenerator
    monkeypatch.setattr(gen, "hidden", types.SimpleNamespace(dynprompt=_Prompt(comp_wired), unique_id="5"),
                        raising=False)
    W, H = size
    args = dict(images=torch.zeros((1, 1, 1, 3)).expand(20, H, W, 3), fps=24.0, first_frame=1001,
                quality="Best (v12, 2 LTX passes)", render="full shot",
                picks=json.dumps({"target": {"frame": frame, "x": 100, "y": 100}}), filename_prefix="t/shot",
                format="png16", ltx_model="ltx.safetensors", alpha_lora="alpha.safetensors",
                text_encoder="built-in empty prompt (no text encoder needed)", video_vae="vae.safetensors",
                audio_vae="avae.safetensors", vitmatte="auto-download", chunk=9, overlap=0, seed=1, erode=10,
                dilate=15, near_px=15, stabilize_radius=2, sam3_threshold=0.3, fast_attention=False, **kw)
    if edge_colour is not None:
        args["edge_colour"] = edge_colour
    res = gen.execute(**args)
    return res, res.expand


def _of(graph, kind):
    return [(k, n) for k, n in graph.items() if n["class_type"] == kind]


def test_graph_off_is_v041(pack, monkeypatch):
    res, graph = _gen_graph(monkeypatch, "off", comp_wired=True)
    assert not _of(graph, "ComfyProAlpha_EdgeColour")
    (fid, final), = _of(graph, "ComfyProAlpha_Final")
    assert final["inputs"]["comp"] is True
    assert res.result[1] == [fid, 1]


def test_graph_quick_adds_edge_colour(pack, monkeypatch):
    res, graph = _gen_graph(monkeypatch, "quick", comp_wired=True, frame=4)
    (eid, ec), = _of(graph, "ComfyProAlpha_EdgeColour")
    (fid, final), = _of(graph, "ComfyProAlpha_Final")
    assert ec["inputs"]["images"] == ["1", 0] and ec["inputs"]["start"] == 4
    assert ec["inputs"]["alpha"] == [fid, 0] and ec["inputs"]["info"] == [fid, 2]
    assert ec["inputs"]["mode"] == "quick" and ec["inputs"]["comp"] is True
    assert final["inputs"]["comp"] is False
    assert res.result[1] == [eid, 0]


def test_graph_default_is_quick(pack, monkeypatch):
    _, graph = _gen_graph(monkeypatch)  # an old workflow without the widget
    assert len(_of(graph, "ComfyProAlpha_EdgeColour")) == 1


def test_edge_colour_input_schema(pack):
    gen = sys.modules["cpa_pack.node"].ComfyProAlphaGenerator
    inp = {i.id: i for i in gen.define_schema().inputs}["edge_colour"]
    assert inp.options == ["quick", "perfect", "off"] and inp.default == "quick" and inp.optional


def test_edge_colour_tooltip_states_proxy_and_cost(pack):
    gen = sys.modules["cpa_pack.node"].ComfyProAlphaGenerator
    tip = {i.id: i for i in gen.define_schema().inputs}["edge_colour"].tooltip
    assert "tiles" not in tip and "proxy" in tip
    assert "a frame at 4K" in tip


# ------------------------------------------------------------------ v15 perfect: clean plate pass

def test_ltx_chunks_default_decode_unchanged(pack):
    from comfy_execution.graph_utils import GraphBuilder
    node = sys.modules["cpa_pack.node"]
    g = GraphBuilder()
    stack = {**STACK, "latent": _link(20)}
    node._ltx_chunks(g, stack, _link(1), 0, 60, [0, 40], 49, "{}")
    kinds = [n["class_type"] for n in g.finalize().values()]
    assert kinds.count("ComfyProAlpha_DecodeAlpha") == 2 and kinds.count("ComfyProAlpha_Stitch") == 1
    assert "ComfyProAlpha_DecodeStitchRGB" not in kinds


def test_ltx_chunks_rgb_decodes_colour(pack):
    from comfy_execution.graph_utils import GraphBuilder
    node = sys.modules["cpa_pack.node"]
    g = GraphBuilder()
    node._ltx_chunks(g, {**STACK, "latent": _link(20)}, _link(1), 0, 60, [0, 40], 49, "{}", decode="rgb")
    kinds = [n["class_type"] for n in g.finalize().values()]
    assert kinds.count("ComfyProAlpha_DecodeStitchRGB") == 1
    assert "ComfyProAlpha_DecodeAlpha" not in kinds and "ComfyProAlpha_Stitch" not in kinds


def test_decode_stitch_rgb_crossfades_to_uint8(pack, monkeypatch):
    import nodes

    class Dec:
        def decode(self, vae, samples, *a):
            return (torch.full((49, 64, 96, 3), samples["v"]),)

    monkeypatch.setattr(nodes, "VAEDecodeTiled", Dec)
    plan = json.dumps({"n": 60, "h": 60, "w": 90, "chunk": 49, "starts": [0, 11]})
    out = pack.I.DecodeStitchRGB.execute(plan=plan, vae=None, chunk_0={"v": 0.2}, chunk_1={"v": 0.2}).result[0]
    assert out.dtype == torch.uint8 and tuple(out.shape) == (60, 60, 90, 3)
    assert int(out[30, 5, 5, 0]) == round(0.2 * 255)


def test_decode_stitch_rgb_streams_frames(pack, monkeypatch):
    """A full shot must not sit in RAM as float32 RGB: only the frames still being cross-faded are kept in float."""
    import nodes
    E = sys.modules["cpa_pack._engine"]
    L, n, h, w = 49, 200, 6, 8
    starts = [0, 40, 80, 120, 151]

    class Dec:
        def decode(self, vae, samples, *a):
            return (torch.full((L, h, w, 3), samples["v"]),)

    big = []
    zeros, empty = torch.zeros, torch.empty

    def spy(real):
        def f(*shape, **kw):
            t = real(*shape, **kw)
            if t.is_floating_point() and t.numel() > (L + 8) * h * w * 3:
                big.append(tuple(t.shape))
            return t
        return f

    monkeypatch.setattr(nodes, "VAEDecodeTiled", Dec)
    monkeypatch.setattr(torch, "zeros", spy(zeros))
    monkeypatch.setattr(torch, "empty", spy(empty))
    v = [0.1, 0.3, 0.5, 0.7, 0.9]
    plan = json.dumps({"n": n, "h": h, "w": w, "chunk": L, "starts": starts})
    out = pack.I.DecodeStitchRGB.execute(plan=plan, vae=None,
                                         **{f"chunk_{i}": {"v": x} for i, x in enumerate(v)}).result[0]
    assert not big, big
    assert out.dtype == torch.uint8 and tuple(out.shape) == (n, h, w, 3)
    for f in (0, 45, 85, 130, 199):
        ws = [(E.chunk_weight(i, starts, L, f - s), v[i]) for i, s in enumerate(starts) if 0 <= f - s < L]
        want = round(sum(wt * x for wt, x in ws) / sum(wt for wt, _ in ws) * 255)
        assert abs(int(out[f, 0, 0, 0]) - want) <= 1, f


def _with_clean_lora(monkeypatch, present=True):
    import folder_paths
    real = folder_paths.get_filename_list
    name = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
    monkeypatch.setattr(folder_paths, "get_filename_list",
                        lambda kind: (real(kind) if kind != "loras" else
                                      [n for n in real(kind) if "clean-plate" not in n] + ([name] if present else [])))
    return name


def test_perfect_without_lora_runs_quick(pack, monkeypatch):
    _with_clean_lora(monkeypatch, present=False)
    gen = sys.modules["cpa_pack.node"].ComfyProAlphaGenerator
    _, graph = _gen_graph(monkeypatch, "perfect", clean_plate_lora="MISSING: ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors")
    (eid, ec), = _of(graph, "ComfyProAlpha_EdgeColour")
    assert ec["inputs"]["mode"] == "perfect" and "clean_plate" not in ec["inputs"]
    assert not _of(graph, "ComfyProAlpha_DecodeStitchRGB")
    problems = gen.validate_inputs(picks=json.dumps({"target": {"frame": 0}}), ltx_model="x", alpha_lora="x",
                                   clean_plate_lora="MISSING: x")
    assert "clean" not in str(problems).lower()


def test_perfect_with_lora_wires_clean_plate(pack, monkeypatch):
    name = _with_clean_lora(monkeypatch)
    _, graph = _gen_graph(monkeypatch, "perfect", clean_plate_lora=name)
    (eid, ec), = _of(graph, "ComfyProAlpha_EdgeColour")
    (did, _), = _of(graph, "ComfyProAlpha_DecodeStitchRGB")
    assert ec["inputs"]["clean_plate"] == [did, 0]
    loras = [n["inputs"]["lora_name"] for _, n in _of(graph, "LTXICLoRALoaderModelOnly")]
    assert name in loras and len(loras) == 2


def test_quick_ignores_clean_lora(pack, monkeypatch):
    name = _with_clean_lora(monkeypatch)
    _, graph = _gen_graph(monkeypatch, "quick", clean_plate_lora=name)
    assert not _of(graph, "ComfyProAlpha_DecodeStitchRGB")
    assert len(_of(graph, "LTXICLoRALoaderModelOnly")) == 1


def test_edge_colour_perfect_uses_clean_plate(pack, out_dirs):
    E = sys.modules["cpa_pack._engine"]
    n, h, w = 2, 64, 96
    a = np.clip((np.arange(w, dtype=np.float32) - 30) / 20, 0, 1)[None].repeat(h, 0)
    B = np.float32([0.1, 0.8, 0.1])
    F = np.where((np.arange(w) < 50)[None, :, None], np.float32([0.9, 0.1, 0.1]), np.float32([0.4, 0.1, 0.1]))
    F = np.broadcast_to(F, (h, w, 3)).astype(np.float32)  # strands lighter than the inside: quick cannot know
    I = E.linear_to_srgb(a[..., None] * F + (1 - a[..., None]) * B)
    args = dict(images=torch.from_numpy(np.stack([I] * n)), start=0, alpha=torch.from_numpy(np.stack([a] * n)),
                info="{}", filename_prefix="t/p", format="png16", first_frame=1, comp=False)
    clean = torch.from_numpy((np.broadcast_to(E.linear_to_srgb(B), (n, h, w, 3)) * 255).round().astype(np.uint8))
    import cv2
    band = (a >= 0.3) & (a <= 0.7)

    def err():
        img = cv2.imread(str(out_dirs / "out" / "t" / "p" / "rgba_00001.png"), cv2.IMREAD_UNCHANGED)
        got = E.srgb_to_linear(img[..., 2::-1].astype(np.float32) / 65535)
        return np.abs(got[band] - F[band]).max()

    pack.I.EdgeColour.execute(mode="quick", **args)
    quick_err = err()  # quick extends the inside colour (0.4 red) over the lighter strands
    per = pack.I.EdgeColour.execute(mode="perfect", clean_plate=clean, **args)
    assert per.ui["pro_alpha"][0]["edge_colour"] == "perfect" and per.ui["pro_alpha"][0]["edge_note"] == ""
    assert err() < 0.02 and quick_err > 0.2


def test_edge_colour_perfect_scales_proxy_inputs(pack, out_dirs):
    """v14 big plate: alpha and plate at native size, clean plate and others on the half-size proxy."""
    import cv2
    E = sys.modules["cpa_pack._engine"]
    n, h, w = 2, 64, 96
    a = np.clip((np.arange(w, dtype=np.float32) - 30) / 20, 0, 1)[None].repeat(h, 0)
    B = np.float32([0.1, 0.8, 0.1])
    F = np.where((np.arange(w) < 50)[None, :, None], np.float32([0.9, 0.1, 0.1]), np.float32([0.4, 0.1, 0.1]))
    F = np.broadcast_to(F, (h, w, 3)).astype(np.float32)
    I = E.linear_to_srgb(a[..., None] * F + (1 - a[..., None]) * B)
    clean = torch.from_numpy((np.broadcast_to(E.linear_to_srgb(B), (n, h // 2, w // 2, 3)) * 255).round()
                             .astype(np.uint8))
    others = torch.zeros(n, h // 2, w // 2)
    others[:, :, :4] = 1  # someone at the left edge, away from the band
    pack.I.EdgeColour.execute(images=torch.from_numpy(np.stack([I] * n)), start=0,
                              alpha=torch.from_numpy(np.stack([a] * n)), info="{}", mode="perfect",
                              filename_prefix="t/m", format="png16", first_frame=1, comp=False,
                              clean_plate=clean, others=others)
    img = cv2.imread(str(out_dirs / "out" / "t" / "m" / "rgba_00001.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (h, w, 4)
    got = E.srgb_to_linear(img[..., 2::-1].astype(np.float32) / 65535)
    band = (a >= 0.3) & (a <= 0.7)
    assert np.abs(got[band] - F[band]).max() < 0.03


def test_edge_colour_radius_scale_ignores_orientation(pack, out_dirs, monkeypatch):
    """A portrait 1080x1920 plate has the same pixel scale as a landscape 1920x1080 one."""
    E = sys.modules["cpa_pack._engine"]
    seen = []
    real = E.quick_colour
    monkeypatch.setattr(E, "quick_colour", lambda I, a, s=1.0, **kw: seen.append(s) or real(I, a, s, **kw))
    for h, w in ((54, 96), (96, 54)):
        pack.I.EdgeColour.execute(images=torch.rand(1, h, w, 3), start=0, alpha=torch.rand(1, h, w), info="{}",
                                  mode="quick", filename_prefix="t/o", format="png16", first_frame=1, comp=False)
    assert seen[0] == seen[1]


# ------------------------------------------------------------------ output paths stay under ComfyUI/output

@pytest.mark.parametrize("prefix", ["../escape", "a/../../escape"])
def test_saves_refuse_prefix_outside_output(pack, out_dirs, prefix):
    with pytest.raises(ValueError, match="output folder"):
        pack.I._save(torch.zeros(1, 8, 8), prefix, "png16", 1)
    args = dict(_edge_args(), filename_prefix=prefix)
    with pytest.raises(ValueError, match="output folder"):
        pack.I.EdgeColour.execute(**args)
    assert not (out_dirs / "escape").exists()
