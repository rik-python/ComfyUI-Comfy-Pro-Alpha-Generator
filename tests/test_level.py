"""v11 level: the isolated-plate LTX pass, re-levelled where the other subjects were painted out."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _engine as E  # noqa: E402


def _scene(painted_level):
    """Target on the left half, another subject on the right half; LTX solid on the target and
    `painted_level` where the other one was painted out."""
    ltx = np.zeros((200, 200), np.float32)
    ltx[20:180, 20:90] = 1.0
    ltx[20:180, 90:180] = painted_level
    ltx[100:110, 30:40] = 0.97  # a soft bit inside the target
    tgt = np.zeros_like(ltx)
    tgt[20:180, 20:90] = 1
    oth = np.zeros_like(ltx)
    oth[20:180, 90:180] = 1
    return ltx, tgt, oth


def test_level_solid_painted_area_no_nan():
    # Painting the neighbour out failed: LTX still mattes it solid (g = 1). Seen on 7 of 34 real passes:
    # 0/0 at every ltx == 1 pixel gave NaN that whited out the layer in Nuke.
    out, g = E.level(*_scene(1.0))
    assert g == 1.0
    assert np.isfinite(out).all()
    assert out[100, 50] == 1.0       # the target stays
    assert out[100, 170] == 0.0      # the painted-out neighbour, far from the target, goes


def test_level_grey_painted_area_still_levelled():
    # The usual case: LTX leaves a grey haze (g) on the painted area; it is levelled away, the target kept.
    out, g = E.level(*_scene(0.4))
    assert abs(g - 0.4) < 1e-6
    assert np.isfinite(out).all()
    assert out[100, 50] == 1.0
    assert out[100, 95] < 0.05
