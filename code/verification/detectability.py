#!/usr/bin/env python3
"""
Computational helpers for the "Null Robustness" project.

Core thesis: an alignment intervention that shifts the outcome distribution from
q (baseline) to p (intervened) produces a *detectable* effect with n samples iff
the statistical divergence between p and q is large enough relative to n. When the
divergence is small (weak intervention, high label noise, strong spurious
correlation masking the causal signal), the experiment yields a NULL result that
reflects experimental power, not model robustness.

This module provides:
  * f-divergences (Hellinger^2, KL, Jensen-Shannon, total variation)
  * The sample-complexity / minimum-detectable-effect characterization
      n*(p,q; delta) ≍ log(1/delta) / h^2(p,q)          (Pensia-Jog-Loh 2024)
  * The reward-overoptimization functional forms (Gao et al. 2022)
      R_bon(d) = d (a - b d),   R_rl(d) = d (a - b log d),   d = sqrt(KL)
  * A Boolean spurious-feature dataset generator (Qiu et al. 2024) exposing the
    parameters (core complexity c, spurious complexity s, correlation lambda).
"""
import numpy as np


# ---------------------------------------------------------------------------
# f-divergences between two discrete distributions p, q (numpy arrays)
# ---------------------------------------------------------------------------
def _clean(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    return p, q

def hellinger2(p, q):
    """Squared Hellinger divergence h^2(p,q) = 1 - sum sqrt(p_i q_i)  in [0,1]."""
    p, q = _clean(p, q)
    return 1.0 - np.sum(np.sqrt(p * q))

def kl(p, q):
    p, q = _clean(p, q)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def jensen_shannon(p, q):
    p, q = _clean(p, q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def total_variation(p, q):
    p, q = _clean(p, q)
    return 0.5 * np.sum(np.abs(p - q))


# ---------------------------------------------------------------------------
# Detectability threshold  (Pensia, Jog, Loh 2024, math.ST)
#   Uniform-prior sample complexity:  n* ≍ log(1/delta) / h^2(p,q)
# ---------------------------------------------------------------------------
def sample_complexity(p, q, delta=0.05):
    """Samples needed to distinguish p from q at Bayes error <= delta."""
    h2 = hellinger2(p, q)
    if h2 <= 0:
        return np.inf
    return np.log(1.0 / delta) / h2

def is_detectable(p, q, n, delta=0.05):
    """Is the p-vs-q effect detectable with n samples at error level delta?"""
    return n >= sample_complexity(p, q, delta)

def min_detectable_divergence(n, delta=0.05):
    """Smallest Hellinger^2 divergence detectable with n samples (necessary cond.)."""
    return np.log(1.0 / delta) / n


# ---------------------------------------------------------------------------
# Reward overoptimization scaling laws  (Gao et al. 2022)
# ---------------------------------------------------------------------------
def gold_reward_bon(d, alpha, beta):
    """Best-of-n gold reward vs. d = sqrt(KL)."""
    return d * (alpha - beta * d)

def gold_reward_rl(d, alpha, beta):
    """RL gold reward vs. d = sqrt(KL)."""
    return d * (alpha - beta * np.log(np.clip(d, 1e-12, None)))


# ---------------------------------------------------------------------------
# Boolean spurious-feature dataset  (Qiu et al. 2024)
#   core feature f_c on c bits, spurious feature f_s on s bits, u useless bits;
#   correlation lambda in [0.5, 1]. Label y = f_c(x_c).
# ---------------------------------------------------------------------------
def parity(bits):
    return np.prod(bits, axis=-1)

def threshold_staircase(bits):
    # sc_d(x) = sign( x1 + x1 x2 + ... + x1..xd )
    d = bits.shape[-1]
    acc = np.zeros(bits.shape[0])
    run = np.ones(bits.shape[0])
    for i in range(d):
        run = run * bits[:, i]
        acc = acc + run
    return np.where(acc >= 0, 1, -1)

def make_boolean_spurious(n_samples, c=3, s=1, u=4, lam=0.9,
                          feat=parity, rng=None):
    """Return (X, y, group) with parametric core/spurious complexity & correlation.

    group == 1 : majority (f_c == f_s), group == 0 : minority (f_c != f_s).
    """
    rng = rng or np.random.default_rng(0)
    n = c + s + u
    X = rng.choice([-1, 1], size=(n_samples, n))
    fc = feat(X[:, :c])
    # enforce correlation: with prob lam make spurious agree with core
    agree = rng.random(n_samples) < lam
    fs = np.where(agree, fc, -fc)
    # embed a spurious feature that computes fs on its own bits
    # (resample spurious bits so that feat(x_s) == fs)
    xs = X[:, c:c + s].copy()
    cur = feat(xs)
    flip = cur != fs
    xs[flip, 0] *= -1              # flipping one bit flips parity/staircase sign
    X[:, c:c + s] = xs
    y = fc
    group = (fc == feat(X[:, c:c + s])).astype(int)
    return X, y, group


if __name__ == "__main__":
    print("== Detectability demo ==")
    # Two Bernoulli outcome distributions: baseline vs. weakly-aligned model.
    for eps in [0.30, 0.10, 0.03, 0.01]:
        q = np.array([0.5, 0.5])
        p = np.array([0.5 + eps, 0.5 - eps])
        h2 = hellinger2(p, q)
        n = sample_complexity(p, q, delta=0.05)
        print(f"effect eps={eps:>4}:  h^2={h2:.5f}  ->  n*≈{n:8.0f} samples "
              f"(TV={total_variation(p,q):.3f}, JS={jensen_shannon(p,q):.5f})")

    print("\nWith n=1000 samples, minimum detectable h^2 =",
          round(min_detectable_divergence(1000), 5),
          "-> weaker interventions look like NULL results.")

    print("\n== Boolean spurious dataset demo ==")
    X, y, g = make_boolean_spurious(2000, c=3, s=1, u=4, lam=0.9)
    print("shape", X.shape, "majority frac", g.mean().round(3),
          "label balance", (y == 1).mean().round(3))
