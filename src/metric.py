"""FREUID Score — local reimplementation for offline model selection.

Positive class = attack / fraud (label 1). Higher score => more likely fraud.

Definitions (from the competition description)
----------------------------------------------
  BPCER(t) = P(score >= t | bona-fide)   # genuine wrongly flagged as attack
  APCER(t) = P(score <  t | attack)      # attack wrongly accepted as genuine

The DET curve is APCER vs BPCER as the threshold t sweeps. Both error rates are
in [0, 1].

  AuDET            = area under the DET curve (linear axes).
                     With x = BPCER, y = APCER this equals (1 - ROC_AUC) for the
                     attack-positive class, which is bounded in [0, 1] and lower
                     is better — consistent with the stated "[0,1], lower better".
  APCER@1%BPCER    = APCER at the threshold where BPCER == 0.01.

  g_audet = 1 - AuDET
  g_apcer = 1 - APCER@1%BPCER
  FREUID  = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)   # lower is better

NOTE ON AuDET AXES
------------------
The brief says AuDET is bounded in [0,1]; that uniquely fits *linear-axis* area
(= 1 - ROC_AUC). Classic NIST DET plots use a normal-deviate (probit) axis whose
area is unbounded, so that cannot be the intended definition. We default to the
linear definition and expose `audet_probit()` only for diagnostic comparison.
Re-confirm against any official metric code before trusting absolute values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def _as_arrays(y_true, y_score):
    y = np.asarray(y_true).astype(float).ravel()
    s = np.asarray(y_score).astype(float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_score {s.shape}")
    if not np.isin(np.unique(y), [0.0, 1.0]).all():
        raise ValueError("y_true must contain only {0,1}")
    if not np.isfinite(s).all():
        raise ValueError("y_score contains non-finite values")
    return y, s


def audet(y_true, y_score) -> float:
    """Area under DET curve (linear axes) = 1 - ROC_AUC. Lower is better."""
    y, s = _as_arrays(y_true, y_score)
    return float(1.0 - roc_auc_score(y, s))


def apcer_at_bpcer(y_true, y_score, bpcer_target: float = 0.01) -> float:
    """APCER at the threshold where BPCER == `bpcer_target`.

    Threshold is the (1 - bpcer_target) quantile of bona-fide scores: the value
    at/above which exactly `bpcer_target` of genuine docs fall. APCER is then the
    fraction of attack scores strictly below that threshold (i.e. accepted).
    Quantile is interpolated so the operating point is hit precisely.
    """
    y, s = _as_arrays(y_true, y_score)
    neg = s[y == 0]            # bona-fide
    pos = s[y == 1]            # attack
    if neg.size == 0 or pos.size == 0:
        raise ValueError("need both classes present")
    # threshold t s.t. P(neg >= t) = bpcer_target  ->  t = quantile(neg, 1-bpcer)
    t = np.quantile(neg, 1.0 - bpcer_target, method="linear")
    apcer = float(np.mean(pos < t))
    return apcer


@dataclass
class FreuidResult:
    freuid: float
    audet: float
    apcer_at_1pct_bpcer: float
    g_audet: float
    g_apcer: float
    roc_auc: float


def freuid_score(y_true, y_score, bpcer_target: float = 0.01) -> FreuidResult:
    """Full FREUID score breakdown. `freuid` is the leaderboard value (lower=better)."""
    a = audet(y_true, y_score)
    p = apcer_at_bpcer(y_true, y_score, bpcer_target)
    g_a = 1.0 - a
    g_p = 1.0 - p
    denom = g_a + g_p
    harm = 0.0 if denom == 0 else 2.0 * g_a * g_p / denom
    return FreuidResult(
        freuid=float(1.0 - harm),
        audet=a,
        apcer_at_1pct_bpcer=p,
        g_audet=g_a,
        g_apcer=g_p,
        roc_auc=1.0 - a,
    )


def audet_probit(y_true, y_score, eps: float = 1e-6) -> float:
    """DIAGNOSTIC ONLY: trapezoidal DET area on normal-deviate (probit) axes.

    Not bounded in [0,1]; provided to compare against the linear definition if a
    future official metric turns out to use probit axes. Do not use for ranking
    unless confirmed.
    """
    from scipy.stats import norm
    y, s = _as_arrays(y_true, y_score)
    neg = np.sort(s[y == 0])
    pos = np.sort(s[y == 1])
    thr = np.unique(np.concatenate([neg, pos]))
    bpcer = np.array([(neg >= t).mean() for t in thr])
    apcer = np.array([(pos < t).mean() for t in thr])
    bpcer = np.clip(bpcer, eps, 1 - eps)
    apcer = np.clip(apcer, eps, 1 - eps)
    x = norm.ppf(bpcer)
    yv = norm.ppf(apcer)
    order = np.argsort(x)
    return float(np.trapz(yv[order], x[order]))


# --- self tests --------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n = 20000

    # 1) Perfect separation: AuDET ~ 0, APCER@1%BPCER ~ 0, FREUID ~ 0.
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1e-3, n)
    r = freuid_score(y, s)
    assert r.audet < 1e-3, r
    assert r.apcer_at_1pct_bpcer < 1e-3, r
    assert r.freuid < 1e-3, r

    # 2) Random scores: AuDET ~ 0.5, APCER@1%BPCER ~ 0.99, FREUID near ~0.66.
    s = rng.random(n)
    r = freuid_score(y, s)
    assert abs(r.audet - 0.5) < 0.02, r
    assert r.apcer_at_1pct_bpcer > 0.95, r

    # 3) AuDET == 1 - sklearn ROC_AUC, exactly.
    s = rng.normal(0, 1, n) + 0.7 * y
    assert abs(audet(y, s) - (1 - roc_auc_score(y, s))) < 1e-12

    # 4) Operating-point sanity: BPCER target actually ~1% of bona-fide above thr.
    neg = s[y == 0]
    t = np.quantile(neg, 0.99, method="linear")
    realized_bpcer = (neg >= t).mean()
    assert abs(realized_bpcer - 0.01) < 0.005, realized_bpcer

    # 5) Inverting scores must make a good model bad (monotonic sanity).
    good = freuid_score(y, s).freuid
    bad = freuid_score(y, -s).freuid
    assert bad > good

    # 6) Better model -> lower FREUID (separation strength ordering).
    f_strong = freuid_score(y, rng.normal(0, 1, n) + 2.0 * y).freuid
    f_weak = freuid_score(y, rng.normal(0, 1, n) + 0.3 * y).freuid
    assert f_strong < f_weak, (f_strong, f_weak)

    print("metric self-test: all passed")
    print(f"  perfect  -> FREUID~{freuid_score(y, y + rng.normal(0,1e-3,n)).freuid:.4f}")
    print(f"  strong   -> FREUID={f_strong:.4f}")
    print(f"  weak     -> FREUID={f_weak:.4f}")
    print(f"  random   -> FREUID={freuid_score(y, rng.random(n)).freuid:.4f}")


if __name__ == "__main__":
    _selftest()
