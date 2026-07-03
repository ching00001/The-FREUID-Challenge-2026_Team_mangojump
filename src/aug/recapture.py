"""Realistic print-and-capture (recapture) augmentation.

Phase-1 diagnosis: the baseline rides a digital generation artifact that is
near-perfect on digital data (even cross-template) but collapses on real
print-and-capture. This module simulates the "analog hole" so training can
force the model OFF that fragile artifact and onto cues that survive capture.

It is heavier / more physical than the light Phase-1 augmentation and models
the print->display/paper->camera channel: resampling, halftone/moire, ink/
screen texture, illumination gradient + specular glare, perspective, white
balance / color-profile shift, chromatic aberration, sensor noise, and double
JPEG. Strength is controlled by a single `strength` in [0,1] so it can be
CALIBRATED to reproduce the public-LB difficulty (baseline ~0.269) and used as
a fast local proxy.

Pure numpy/cv2 (no albumentations dep). All ops operate on uint8 RGB HxWx3.
"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(0)


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


# cache coordinate grids per (h,w) — recreating them per image dominated cost
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _grids(h: int, w: int):
    key = (h, w)
    g = _GRID_CACHE.get(key)
    if g is None:
        yy, xx = np.mgrid[0:h, 0:w]
        g = (yy.astype(np.float32), xx.astype(np.float32))
        _GRID_CACHE[key] = g
    return g


# --- individual physical effects --------------------------------------------
def resample(img, scale, rng):
    h, w = img.shape[:2]
    interp_down = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=interp_down)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def halftone_moire(img, period, angle_deg, amp, rng):
    """Additive sinusoidal grid -> screen/print halftone & moire interference."""
    h, w = img.shape[:2]
    yy, xx = _grids(h, w)
    a = np.deg2rad(angle_deg)
    grid = np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / period)
    grid += np.sin(2 * np.pi * (xx * np.cos(a + 1.2) + yy * np.sin(a + 1.2)) / (period * 0.93))
    pat = (grid * amp)[..., None]
    return np.clip(img.astype(np.float32) + pat, 0, 255).astype(np.uint8)


def illumination_glare(img, grad_strength, glare_strength, rng):
    """Low-frequency lighting gradient + an elliptical specular highlight."""
    h, w = img.shape[:2]
    yy, xx = _grids(h, w)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    grad = (gx * (xx / w - 0.5) + gy * (yy / h - 0.5))
    field = 1.0 + grad_strength * grad
    out = img.astype(np.float32) * field[..., None]
    if glare_strength > 0:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        rx, ry = rng.uniform(0.1, 0.3) * w, rng.uniform(0.1, 0.3) * h
        spot = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        out += (glare_strength * 255.0) * spot[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def perspective(img, jitter, rng):
    h, w = img.shape[:2]
    d = jitter * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)


def white_balance(img, shift, rng):
    gains = 1.0 + rng.uniform(-shift, shift, 3).astype(np.float32)
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def chromatic_aberration(img, px, rng):
    if px < 1:
        return img
    b, g, r = cv2.split(img)
    sx, sy = int(rng.integers(-px, px + 1)), int(rng.integers(-px, px + 1))
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    r = cv2.warpAffine(r, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT101)
    return cv2.merge([b, g, r])


def sensor_noise(img, sigma, rng):
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def blur(img, k):
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0) if k >= 3 else img


# --- composed pipeline -------------------------------------------------------
def recapture(img, strength: float = 0.6, p: float = 1.0, rng=None):
    """Apply the full print-and-capture channel at a given strength in [0,1].

    `p` is the probability the whole pipeline is applied (else returns img).
    Each sub-effect also samples its own randomness so identical strength still
    yields diverse outputs. strength scales magnitudes monotonically.
    """
    rng = _rng(rng)
    if rng.random() > p:
        return img
    s = float(np.clip(strength, 0, 1))
    out = img

    # 1) double-resample (print then camera downsample)
    out = resample(out, scale=rng.uniform(0.4, 0.8) - 0.25 * s, rng=rng)
    # 2) optical blur
    out = blur(out, k=rng.integers(1, int(2 + 5 * s) | 1 + 1))
    # 3) halftone / moire
    if rng.random() < 0.7 * (0.4 + s):
        out = halftone_moire(out, period=rng.uniform(2.5, 6.0),
                             angle_deg=rng.uniform(0, 180),
                             amp=rng.uniform(3, 6 + 14 * s), rng=rng)
    # 4) illumination + glare
    out = illumination_glare(out, grad_strength=rng.uniform(0.05, 0.15 + 0.35 * s),
                             glare_strength=rng.uniform(0, 0.15 + 0.35 * s), rng=rng)
    # 5) geometric perspective
    out = perspective(out, jitter=rng.uniform(0, 0.01 + 0.05 * s), rng=rng)
    # 6) color: white balance + chromatic aberration
    out = white_balance(out, shift=rng.uniform(0.02, 0.05 + 0.15 * s), rng=rng)
    out = chromatic_aberration(out, px=int(rng.integers(0, 1 + int(3 * s))), rng=rng)
    # 7) sensor noise
    out = sensor_noise(out, sigma=rng.uniform(1, 2 + 10 * s), rng=rng)
    # 8) double JPEG (print pipeline + capture re-encode)
    out = jpeg(out, q=int(rng.integers(60, 90)))
    out = jpeg(out, q=int(rng.integers(max(20, 60 - int(40 * s)), 75)))
    return np.ascontiguousarray(out)


# named strength presets for calibration / eval profiles
PRESETS = {
    "rc0.3": 0.3, "rc0.5": 0.5, "rc0.7": 0.7, "rc0.9": 0.9, "rc1.0": 1.0,
}
