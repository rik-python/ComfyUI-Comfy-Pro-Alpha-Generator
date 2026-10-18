"""v15 edge colour: sRGB curves, quick foreground colour (Blur-Fusion + texture extend), temporal smoothing,
clean-plate unmix and its confidence gate."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _engine as E  # noqa: E402


def _scene(h=64, w=96, fg=(0.9, 0.1, 0.1), bg=(0.1, 0.8, 0.1)):
    """Linear plate of a flat red subject over flat green: soft alpha ramp, solid core on the right."""
    a = np.clip((np.arange(w, dtype=np.float32) - 30) / 20, 0, 1)[None].repeat(h, 0)
    F = np.broadcast_to(np.float32(fg), (h, w, 3)).copy()
    B = np.broadcast_to(np.float32(bg), (h, w, 3)).copy()
    return a[..., None] * F + (1 - a[..., None]) * B, a, F, B


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.abs(E.linear_to_srgb(E.srgb_to_linear(x)) - x).max() < 1e-5
    assert abs(float(E.srgb_to_linear(np.float32(0.5))) - 0.21404) < 1e-4


def test_quick_recovers_foreground_colour_in_band():
    I, a, F, _ = _scene()
    est = E.quick_colour(I, a, s=1.0)  # the paper's radii (90, 6); Blur-Fusion is approximate, not exact
    band = (a > 0.2) & (a < 0.95)
    err = np.abs(est[band] - F[band])
    assert err.max() < 0.1 and err.mean() < 0.04          # off (the plate): max 0.6, mean 0.21
    assert np.abs(I[band] - F[band]).max() > 0.3          # sanity: the scene really is contaminated


def _deep(a, px=3):
    """Solid pixels more than px away from any alpha < 0.98 (the inside quick never touches)."""
    import cv2
    return cv2.distanceTransform((a >= 0.98).astype(np.uint8), cv2.DIST_L2, 3) > px


def test_quick_core_equals_plate():
    I, a, _, _ = _scene()
    est = E.quick_colour(I, a, s=0.05)
    core = _deep(a)
    assert np.array_equal(est[core], I[core])


def test_quick_cleans_rim_where_alpha_runs_wide():
    # LTX edges often run a pixel wide: alpha says solid, the plate still shows the background there
    I, _, F, B = _scene()
    a = np.zeros(I.shape[:2], np.float32)
    a[:, 50:] = 1
    I = np.where((np.arange(I.shape[1]) < 52)[None, :, None], B, F).astype(np.float32)  # cols 50-51: grass
    est = E.quick_colour(I, a, s=1.0)
    assert np.abs(est[:, 50:52] - F[:, 50:52]).max() < 0.05      # off would be the green background
    assert np.array_equal(est[:, 60:], I[:, 60:])


def test_quick_hard_alpha_no_nan():
    I, _, _, _ = _scene()
    a = np.zeros(I.shape[:2], np.float32)
    a[:, 50:] = 1
    est = E.quick_colour(I, a, s=0.05)
    assert np.isfinite(est).all() and np.array_equal(est[:, 53:], I[:, 53:])
    assert np.abs(est[:, 50:53] - I[:, 50:53]).max() < 1e-4  # rim: the same flat colour, rebuilt


def test_quick_border_subject():
    I, a, _, _ = _scene()
    est = E.quick_colour(I[:, ::-1].copy(), a[:, ::-1].copy(), s=0.05)  # core touches the left border
    assert est.shape == I.shape and np.isfinite(est).all()


def test_smooth_colour_reduces_flicker_on_static_scene():
    rng = np.random.default_rng(0)
    I, a, F, _ = _scene()
    Fs = [F + rng.normal(0, 0.05, F.shape).astype(np.float32) for _ in range(9)]
    gray = np.repeat(E.to_gray((I * 255).astype(np.uint8)[None]), 9, 0)
    out = E.smooth_colour(Fs, [a] * 9, gray, radius=2)
    band = (a > 0.002) & (a < 0.999)
    before = np.mean([np.abs(Fs[t] - Fs[t - 1])[band].mean() for t in range(1, 9)])
    after = np.mean([np.abs(out[t] - out[t - 1])[band].mean() for t in range(1, 9)])
    assert after < before / 2


def test_smooth_colour_index_matches_full_run():
    rng = np.random.default_rng(1)
    I, a, F, _ = _scene()
    Fs = [F + rng.normal(0, 0.05, F.shape).astype(np.float32) for _ in range(5)]
    gray = np.repeat(E.to_gray((I * 255).astype(np.uint8)[None]), 5, 0)
    full = E.smooth_colour(Fs, [a] * 5, gray, radius=2)
    assert np.allclose(E.smooth_colour(Fs, [a] * 5, gray, radius=2, index=2), full[2])


def test_smooth_colour_flow_at_reduced_size_still_smooths():
    rng = np.random.default_rng(2)
    I, a, F, _ = _scene(64, 200)
    Fs = [F + rng.normal(0, 0.05, F.shape).astype(np.float32) for _ in range(5)]
    gray = np.repeat(E.to_gray((I * 255).astype(np.uint8)[None]), 5, 0)
    out = E.smooth_colour(Fs, [a] * 5, gray, radius=2, index=2, max_w=100)
    band = (a > 0.002) & (a < 0.999)
    assert out.shape == F.shape
    assert np.abs(out - F)[band].mean() < np.abs(Fs[2] - F)[band].mean() / 1.5


# ------------------------------------------------------------------ perfect: clean plate unmix

def test_unmix_exact_with_true_background():
    I, a, F, B = _scene()
    Fq = E.quick_colour(I, a, s=0.05)
    out = E.unmix_colour(I, a, B, Fq, np.ones_like(a))
    band = (a >= 0.25) & (a <= 0.85)  # full unmix weight; above 0.85 quick takes over (LTX rims run wide)
    assert np.abs(out[band] - F[band]).max() < 1e-4


def test_unmix_falls_back_to_quick_without_confidence():
    I, a, F, B = _scene()
    Fq = E.quick_colour(I, a, s=0.05)
    out = E.unmix_colour(I, a, B * 0 + 0.5, Fq, np.zeros_like(a))
    assert np.array_equal(out, Fq)


def test_unmix_alpha_zero_is_finite():
    I, _, _, B = _scene()
    a = np.zeros(I.shape[:2], np.float32)
    out = E.unmix_colour(I, a, B, I, np.ones_like(a))
    assert np.isfinite(out).all()


def test_confidence_high_when_clean_plate_matches_low_when_not():
    I, a, F, B = _scene()
    assert E.clean_confidence(B, I, a, None, s=0.05)[a > 0.3].mean() > 0.9
    wrong = B.copy()
    wrong[..., 0] += 0.5
    assert E.clean_confidence(wrong, I, a, None, s=0.05)[a > 0.3].mean() < 0.1


def test_confidence_zero_next_to_other_person():
    I, a, F, B = _scene()
    others = np.zeros_like(a)
    others[:, :10] = 1
    c = E.clean_confidence(B, I, a, others, s=0.05)
    assert c[:, :10].max() == 0


def test_match_clean_removes_colour_drift():
    # LTX's clean plate drifts in colour (seen: about -0.04 blue on grass); matched to the real plate at the ring
    I, a, F, B = _scene()
    drift = B + np.float32([0.0, 0.0, -0.04])
    out = E.match_clean(drift, I, a, s=0.05)
    band = a > 0.3
    assert np.abs(out[band] - B[band]).max() < 0.01


def test_confidence_ignores_regenerated_texture():
    # LTX rebuilds grass with different blades: only the low-frequency colour has to agree
    rng = np.random.default_rng(3)
    I, a, F, B = _scene()
    noisy = np.clip(B + rng.normal(0, 0.05, B.shape).astype(np.float32), 0, 1)
    assert E.clean_confidence(noisy, I, a, None, s=1.0)[a > 0.3].mean() > 0.9


def test_confidence_zero_when_subject_not_removed():
    # Clean Plate fails when people fill the frame: it re-renders them. Then B equals the plate inside the
    # subject too, and an unmix would hand the background straight back
    I, a, F, B = _scene()
    assert E.clean_confidence(I.copy(), I, a, None, s=1.0)[(a > 0.05) & (a < 0.95)].max() < 0.05
