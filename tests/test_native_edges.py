"""v14 native-resolution edges: proxy size, tile grid, tile selection, stitch weights, merge."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _engine as E  # noqa: E402


def test_needs_proxy():
    assert not E.needs_proxy(1920, 1080)
    assert not E.needs_proxy(1920, 1088)
    assert E.needs_proxy(1921, 1080)
    assert E.needs_proxy(1920, 1089)
    assert E.needs_proxy(3840, 2160)


def test_needs_proxy_vertical_1080p_unchanged():
    # vertical 1080p ran LTX directly in v0.3.0 (1088x1920): it must keep doing so
    assert not E.needs_proxy(1080, 1920)
    assert not E.needs_proxy(1088, 1920)
    assert E.needs_proxy(1080, 1921)
    assert E.needs_proxy(1089, 1920)
    assert E.needs_proxy(2160, 3840)


def test_proxy_size_vertical_fits_portrait_box():
    assert E.proxy_size(2160, 3840) == (1080, 1920)


def test_tile_grid_vertical_uses_portrait_tiles():
    grid = E.tile_grid(2160, 3840)
    assert all((tw, th) == (1088, 1920) for _, _, tw, th in grid)
    assert len(grid) == 9
    cover = np.zeros((3840, 2160), bool)
    for x, y, tw, th in grid:
        cover[y:y + th, x:x + tw] = True
    assert cover.all()


def test_proxy_size_uhd():
    assert E.proxy_size(3840, 2160) == (1920, 1080)


@pytest.mark.parametrize("w,h", [(2160, 3840), (3840, 1634)])
def test_proxy_size_fits_even_and_keeps_aspect(w, h):
    pw, ph = E.proxy_size(w, h)
    bw, bh = (1920, 1088) if w >= h else (1088, 1920)  # portrait plates fit a portrait box
    assert pw <= bw and ph <= bh
    assert pw % 2 == 0 and ph % 2 == 0
    assert abs(pw / ph - w / h) < 0.01


def test_grid_uhd():
    assert E.grid_axis(3840, 1920) == [0, 960, 1920]
    assert E.grid_axis(2160, 1088) == [0, 536, 1072]
    assert len(E.tile_grid(3840, 2160)) == 9


def test_grid_short_axis():
    assert E.tile_grid(2048, 1080) == [(0, 0, 1920, 1080), (128, 0, 1920, 1080)]


@pytest.mark.parametrize("w,h", [(3840, 2160), (4096, 2160), (6144, 3160), (7680, 4320)])
def test_grid_covers(w, h):
    grid = E.tile_grid(w, h)
    cover = np.zeros((h, w), bool)
    for x, y, tw, th in grid:
        cover[y:y + th, x:x + tw] = True
    assert cover.all()
    xs, ys = E.grid_axis(w, 1920), E.grid_axis(h, 1088)
    assert all(b - a <= 1920 / 2 for a, b in zip(xs, xs[1:]))
    assert all(b - a <= 1088 / 2 for a, b in zip(ys, ys[1:]))
    assert xs[-1] + 1920 == w and ys[-1] + 1088 == h


def test_select_tiles_empty():
    assert E.select_tiles(np.zeros((3, 540, 960), np.float32), E.tile_grid(3840, 2160), (3840, 2160)) == []


def test_select_tiles_left_half():
    sel = np.zeros((3, 540, 960), np.float32)
    sel[:, :, :300] = 1  # one vertical edge at proxy x = 300 (native 1200): only the two left columns touch it
    assert set(E.select_tiles(sel, E.tile_grid(3840, 2160), (3840, 2160))) == {0, 1, 3, 4, 6, 7}


def test_feather_partition():
    W, H = 3840, 2160
    grid = E.tile_grid(W, H)
    wsum = np.zeros((H, W), np.float32)
    for i, (x, y, tw, th) in enumerate(grid):
        f = E.tile_feather(grid, i, (W, H))
        assert f.shape == (th, tw)
        wsum[y:y + th, x:x + tw] += f
        if x == 0:
            assert f[th // 2, 0] == 1.0
        if x + tw == W:
            assert f[th // 2, -1] == 1.0
        if y == 0:
            assert f[0, tw // 2] == 1.0
        if y + th == H:
            assert f[-1, tw // 2] == 1.0
    assert (wsum > 0).all()


def _square(n=200, lo=40, hi=160):
    b = np.zeros((n, n), np.float32)
    b[lo:hi, lo:hi] = 1
    return b


def test_merge_core_is_base():
    base = _square()
    out = E.merge_native(base, np.zeros_like(base), np.ones_like(base), 2.0)
    assert abs(out[100, 100] - 1.0) < 1e-6


def test_merge_no_coverage():
    base = np.random.default_rng(0).random((120, 160)).astype(np.float32)
    tiles = np.random.default_rng(1).random((120, 160)).astype(np.float32)
    out = E.merge_native(base, tiles, np.zeros_like(base), 2.0)
    assert np.allclose(out, base, atol=1e-6)


def test_merge_agree_takes_tiles():
    base = np.full((120, 160), 0.5, np.float32)
    strands = np.zeros_like(base)
    strands[:, ::4] = 0.05
    tiles = base + strands
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0)
    assert np.allclose(out, tiles, atol=1e-6)


def test_merge_others_keep_base():
    base = np.full((120, 160), 0.5, np.float32)
    tiles = base + 0.04
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0, others=np.ones_like(base))
    assert np.allclose(out, base, atol=1e-6)


def test_merge_no_tile_leak_on_other_person():
    # base leaks faintly onto a neighbour (glasses at 0.1), the tile makes it a 0.4 blob: the neighbour keeps base
    base = np.zeros((300, 400), np.float32)
    base[:, :100] = 1  # the subject
    base[140:160, 250:290] = 0.1
    tiles = base.copy()
    tiles[140:160, 250:290] = 0.4
    others = np.zeros_like(base)
    others[60:240, 200:340] = 1  # the neighbour's SAM3 mask
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0, others=others)
    assert abs(out[150, 270] - 0.1) < 1e-6


def test_merge_strands_away_from_others_still_from_tiles():
    base = np.full((300, 400), 0.5, np.float32)
    tiles = base + 0.04
    others = np.zeros_like(base)
    others[:, 350:] = 1  # someone 250+ px away
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0, others=others)
    assert np.allclose(out[:, :200], tiles[:, :200], atol=1e-6)


def _uhd_feathers():
    grid = E.tile_grid(3840, 2160)
    return grid, [E.tile_feather(grid, i, (3840, 2160)) for i in range(len(grid))]


def test_stitch_tiles_constant():
    grid, fe = _uhd_feathers()
    tiles = [np.full((1088, 1920), 0.3, np.float32) for _ in grid]
    a, cov = E.stitch_tiles(tiles, grid, list(range(len(grid))), (3840, 2160), fe)
    assert a.shape == (2160, 3840) and np.allclose(a, 0.3, atol=1e-5)
    assert np.allclose(cov, 1.0, atol=1e-5)


def test_stitch_tiles_partial_fades_into_unrendered():
    grid, fe = _uhd_feathers()
    a, cov = E.stitch_tiles([np.full((1088, 1920), 0.7, np.float32)], grid, [0], (3840, 2160), fe)
    assert abs(a[100, 100] - 0.7) < 1e-5 and abs(cov[100, 100] - 1.0) < 1e-5
    assert cov[2159, 3839] == 0 and a[2159, 3839] == 0
    assert 0 < cov[100, 1700] < 1  # overlap with the unrendered tile 1: fades instead of stepping


def test_stitch_tiles_none():
    grid, fe = _uhd_feathers()
    a, cov = E.stitch_tiles([], grid, [], (3840, 2160), fe)
    assert not a.any() and not cov.any()


def test_merge_no_haze_far_from_subject():
    # a tile's faint haze on someone else (base 0 there, far from the subject) must not reach the output
    base = np.zeros((400, 600), np.float32)
    base[150:250, 50:150] = 1  # the subject
    tiles = base.copy()
    tiles[:, 350:] = 0.15  # haze >200 px away
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0)
    assert out[200, 500] == 0


def test_merge_keeps_strands_next_to_subject():
    # strands just outside the base's soft edge (within ~20 px at 4K) still come from the tiles
    base = np.zeros((400, 600), np.float32)
    base[150:250, 50:150] = 1
    base[150:250, 150:156] = np.linspace(.9, .05, 6)
    tiles = base.copy()
    tiles[180:181, 156:172] = 0.3  # a strand reaching 16 px past the base's edge
    out = E.merge_native(base, tiles, np.ones_like(base), 2.0)
    assert out[180, 165] > 0.2
