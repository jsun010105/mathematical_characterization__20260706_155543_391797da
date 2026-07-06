# Computational Tools

## Installed packages (in project `.venv`)
| Tool | Purpose |
|------|---------|
| sympy | Symbolic manipulation of divergences / sample-complexity formulas |
| numpy, scipy | Numerical divergences, thresholds, statistical tests |
| networkx | Causal-graph construction & complexity measures |
| pypdf | PDF text extraction for reading papers |
| httpx | arXiv / Semantic Scholar API access |

## verification/detectability.py
Self-contained helper implementing the project's mathematical core:

- **f-divergences**: `hellinger2`, `kl`, `jensen_shannon`, `total_variation`.
- **Detectability threshold** (Pensia–Jog–Loh 2024):
  `sample_complexity(p,q,δ) = log(1/δ)/h²(p,q)`, `is_detectable`,
  `min_detectable_divergence(n,δ)`.
- **Reward overoptimization forms** (Gao et al. 2022): `gold_reward_bon`,
  `gold_reward_rl` in terms of `d=√KL`.
- **Boolean spurious dataset** (Qiu et al. 2024): `make_boolean_spurious(
  n, c, s, u, lam, feat)` with parity / threshold-staircase features — parametric
  control of core complexity `c`, spurious complexity `s`, correlation `λ`.

Run: `python code/verification/detectability.py`
Demonstrates numerically that an effect of size ε=0.03 needs ≈6,650 samples and
appears as a NULL result at n=1,000 — mirroring Gao et al.'s ~2,000-comparison
threshold.

## Notes for proof construction
- Use `sympy` to derive closed forms for `h²` under the Boolean dataset's `D_λ`
  mixture and to verify the strong-data-processing contraction coefficients.
- Use `make_boolean_spurious` to empirically trace the null→detectable boundary as
  `(c,s,λ)` vary, then compare against the analytic threshold.
