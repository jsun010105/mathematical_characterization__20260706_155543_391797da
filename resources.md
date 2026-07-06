# Resources Catalog

## Summary
Resources gathered for *Mathematical Characterization of Null Robustness in LLM
Alignment via Synthetic Dataset Topology*. The project is, at its mathematical core,
a **detectability / sample-complexity** problem: null results are hypothesized to
arise when dataset structure shrinks the statistical divergence between aligned and
unaligned outputs below the threshold resolvable at the given sample size.

- Papers downloaded: **14** (see `papers/` and `papers/README.md`)
- Prior results cataloged: **8 theorems/results** (see `literature_review.md`)
- Computational tools set up: **5 packages + 1 verification module**

## Papers
| # | Title | Authors | Year | File | Key result |
|---|-------|---------|------|------|-----------|
| 1 | Sample Complexity of Simple Binary Hypothesis Testing | Pensia, Jog, Loh | 2024 | papers/2403.16981_*.pdf | `n*≍log(1/δ)/h²(p,q)` (detectability) |
| 2 | Optimal Sample Complexity of PAC Learning | Hanneke | 2015 | papers/1507.00473_*.pdf | `n=Θ((d+log 1/δ)/ε)` |
| 3 | Likelihood-based Alternative to NHST | Rafi et al. | 2018 | papers/1806.02419_*.pdf | meaning of a null result |
| 4 | Scaling Laws for Reward Model Overoptimization | Gao, Schulman, Hilton | 2022 | papers/2210.10760_*.pdf | gold-reward forms; ~2000-comparison null threshold |
| 5 | Reward Model Ensembles Mitigate Overoptimization | Coste et al. | 2023 | papers/2310.02743_*.pdf | when overoptimization appears |
| 6 | Rethinking Reward Model Evaluation | Zhang et al. | 2025 | papers/2505.12763_*.pdf | overoptimization-lens evaluation |
| 7 | Direct Preference Optimization | Rafailov et al. | 2023 | papers/2305.18290_*.pdf | closed-form policy↔reward |
| 8 | Complexity Matters (spurious feature dynamics) | Qiu et al. | 2024 | papers/2403.03375_*.pdf | parametric Boolean dataset (c,s,λ) |
| 9 | Clever Hans Mirage (spurious survey) | Ye et al. | 2024 | papers/2402.12715_*.pdf | taxonomy of spurious correlations |
| 10 | Instance-Dependent Label Noise (2nd order) | Zhu, Song, Liu | 2020 | papers/2012.11854_*.pdf | covariance corrections under noise |
| 11 | Identifiability for Causal Representation Learning | Jin et al. | 2023 | papers/2310.15450_*.pdf | 2 uncoupled interventions/node ⇒ recovery |
| 12 | PAC Learnability: Necessary vs Sufficient | Van den Berg et al. | 2025 | papers/2501.08887_*.pdf | VC/compression sufficient ≠ necessary |
| 13 | Neural Scaling Laws Rooted in Data Distribution | Brill et al. | 2024 | papers/2412.07942_*.pdf | scaling exponents from data topology |
| 14 | Modern Mathematics of Deep Learning | Berner et al. | 2021 | papers/2105.04026_*.pdf | approximation/optimization/generalization survey |

## Prior Results Catalog (for proofs)
| Result | Source | Statement | Used for |
|--------|--------|-----------|----------|
| Detectability threshold | Pensia et al. 2024 | `n≍log(1/δ)/h²(p,q)` | master N/S condition |
| PAC threshold | Hanneke 2015 | `n=Θ((d+log 1/δ)/ε)` | learnability separation |
| Overopt. functional forms | Gao et al. 2022 | `R_bon=d(α−βd)`, `R_RL=d(α−β log d)` | alignment detectability |
| DPO reparameterization | Rafailov et al. 2023 | `π*∝π_ref e^{R/β}` | intervention as dist. shift |
| 2nd-order noise correction | Zhu et al. 2020 | covariance restores consistency | label-noise axis |
| Causal identifiability | Jin et al. 2023 | 2 uncoupled interventions/node | causal-complexity axis |
| Sufficient≠Necessary (PAC) | Van den Berg 2025 | VC/compression not necessary | necessity guardrail |
| Scaling from data topology | Brill et al. 2024 | percolation ⇒ power-law exponents | topology→effect precedent |

## Computational Tools
| Tool | Purpose | Location |
|------|---------|----------|
| sympy | symbolic divergence/threshold derivation | .venv (pip) |
| numpy/scipy | numerical divergences & tests | .venv (pip) |
| networkx | causal-graph complexity | .venv (pip) |
| detectability.py | divergences, threshold, Boolean dataset generator | code/verification/detectability.py |

## Resource Gathering Notes

### Search strategy
The paper-finder service was **not running** (localhost:8000 unavailable), so I ran
manual searches via the **arXiv API** and Semantic Scholar (`search_helper.py`) across
the hypothesis's three axes — detectability/statistical power, dataset structure
(spurious correlations, label noise, causal graphs), and alignment interventions
(RLHF/DPO/reward overoptimization) — plus foundational learning theory.

### Selection criteria
Prioritized (1) the **detectability foundation** (sample complexity ↔ divergence),
(2) the **alignment-specific null phenomenon** (Gao et al.'s overoptimization + data
threshold), and (3) a **parametrically controllable synthetic dataset** (Qiu et al.).
Then filled the label-noise, causal-graph, and necessity-of-conditions gaps.

### Challenges
- No paper-finder service → manual API search; arXiv required https + redirect
  following.
- One PDF (1806.02419) needed an explicit version suffix (`v2`) to download.
- PDF page-rendering (poppler) unavailable → read papers via `pypdf` text extraction.

## Recommendations for Proof Construction
1. **Proof strategy.** Anchor on the detectability threshold `n≍log(1/δ)/h²`; prove
   per-axis **divergence-contraction lemmas** (spurious correlation via
   data-processing inequality on the Markov chain; label noise via strong-DPI
   contraction coefficient `η(T)`; causal complexity via interventions-per-node), then
   compose into one N&S condition.
2. **Key prerequisites.** Pensia et al. (threshold), Qiu et al. (dataset + Markov
   structure), Zhu et al. (noise), Jin et al. (causal), with Hanneke as the PAC
   analogue.
3. **Computational tools.** `code/verification/detectability.py` (numeric divergences,
   threshold, dataset) + `sympy` for closed forms of `h²` under `D_λ` and the
   contraction coefficients.
4. **Potential difficulties.** Asymmetric-prior gap (Hellinger characterization can
   break for rare-success alignment events); sufficient≠necessary (prove necessity via
   explicit lower bounds); static-divergence vs. dynamic training-trajectory effects
   (spurious features "not forgotten").
