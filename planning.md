# Planning: Mathematical Characterization of Null Robustness in LLM Alignment via Synthetic Dataset Topology

## Motivation & Novelty Assessment

### Why This Research Matters
Alignment experiments routinely report *null results* — an intervention (RLHF, DPO,
reward modeling, safety fine-tuning) produces no measurable behavioral change — and
these are casually attributed to "model robustness." If instead the null is an
artifact of a dataset that mathematically *cannot* reveal the effect at the sample
size used, then the field is drawing safety conclusions from underpowered
experiments. A rigorous criterion separating **genuine robustness** from
**experimental artifact** is therefore directly safety-relevant.

### Gap in Existing Work
The literature (see `literature_review.md`) contains two disjoint halves:
(a) a tight **detectability threshold** `n ≍ log(1/δ)/h²(p,q)` (Pensia–Jog–Loh 2024),
and (b) three separate **divergence-shrinking mechanisms** — spurious correlation
(Qiu et al. 2024), instance-dependent label noise (Zhu et al. 2020), causal-graph
complexity (Jin et al. 2023). **No result connects the dataset's structural
parameters `(λ, e, k)` to the effective divergence, and hence to the detectability
threshold.** That bridge is the open problem this project targets.

### Our Novel Contribution
We prove that **each dataset-topology axis is a divergence-contracting Markov
channel**, so its effect on detectability is governed by the *strong data-processing
inequality* (SDPI). This yields:
1. An **exact/local contraction coefficient** per axis: label noise `η=(1−2e)²`,
   spurious-mediated measurement `η=(2λ−1)²`, causal chain of length `k` gives
   `η=η₀^k` (exponential in path length).
2. A **composition theorem** (multiplicative contraction along the composed channel)
   giving one effective threshold
   `n* ≍ log(1/δ) / ( η_noise · η_spur · η_causal · h²_ideal )`.
3. A **necessary-and-sufficient** null-result characterization (up to universal
   constants) and a **diagnostic decision rule** distinguishing robustness from
   artifact.

### Experiment (Computational Verification) Justification
- **V1 — BSC Hellinger contraction:** confirm the derived `(1−2e)²` factor against the
  *exact* Hellinger divergence across effect sizes (checks the local approximation is
  faithful in the near-null regime that matters).
- **V2 — Channel composition:** confirm `η(K₂∘K₁) ≤ η(K₂)η(K₁)` numerically, and
  chain dilution `η₀^k`, validating the composition theorem.
- **V3 — Synthetic-dataset boundary:** on the Qiu et al. Boolean dataset, trace the
  empirical null→detectable boundary as `λ` varies and check it matches the predicted
  `n ≍ log(1/δ)/((2λ−1)² h²)`.
- **V4 — Threshold curve:** verify `n*` vs `h²_eff` and the min-detectable-divergence
  floor used by the diagnostic.

## Research Question
Do null results in alignment experiments arise from identifiable mathematical
properties of dataset structure (feature-correlation topology, label-noise
distribution, causal-graph complexity) rather than intrinsic model robustness — and
can we state necessary and sufficient conditions under which an intervention produces
a detectable effect?

## Background and Motivation
See Motivation section above and `literature_review.md`. Core reduction: an
intervention shifting the outcome distribution `q → p` is detectable at sample size
`n`, error `δ`, iff `h²(p,q) ≳ log(1/δ)/n` (Pensia et al.). Dataset structure enters
by mapping the *ideal* shift through an observation channel that contracts `h²`.

## Hypothesis Decomposition
- **H1 (detectability = divergence vs. n):** restate Pensia threshold for alignment shifts.
- **H2 (topology contracts divergence):** each of noise/correlation/causal-complexity is
  a channel; prove contraction coefficient per axis.
- **H3 (synthetic control):** the Boolean dataset dials `λ` and reproduces the boundary.

## Proposed Methodology
### Approach
Anchor on the detectability threshold; model each dataset property as a Markov kernel;
apply DPI (qualitative) and SDPI (quantitative, with explicit η); compose; prove
matching lower bound for necessity; instantiate on the synthetic testbed.

### Proof Steps
1. **Thm 1** master detectability inequality (cite + restate Pensia).
2. **Lemma 2** DPI for `h²`: any post-processing channel cannot increase `h²`.
3. **Lemma 3** label-noise contraction `η=(1−2e)²` (exact for χ², local for h²; verified).
4. **Lemma 4** spurious-measurement contraction `η=(2λ−1)²` (BSC with crossover `1−λ`).
5. **Lemma 5** causal-chain contraction `η₀^k` (SDPI multiplicativity along a chain).
6. **Thm 6** composition → effective threshold; **Thm 7** N&S null characterization.
7. **Thm 8** diagnostic decision rule (robustness vs artifact) via the `h²_floor=log(1/δ)/n`.

### Baselines / Comparison
Compare predicted thresholds against the exact numerical divergences and against the
empirical detection boundary on the synthetic dataset.

### Evaluation Metrics
Exact vs. predicted `h²` ratio (contraction factor error), predicted vs. empirical
detection sample size, and Cohen's-d / Bayes-factor sanity on the synthetic runs.

### Statistical Analysis Plan
Two-sample detection via likelihood-ratio / divergence estimate; `δ=0.05`; report
sample-complexity curves and the boundary crossing. Multiple λ, e, k grids; seeds fixed.

## Expected Outcomes
Contraction factors match exact divergences within the near-null regime; empirical
boundary tracks the predicted curve; the diagnostic correctly flags underpowered nulls.

## Timeline and Milestones
Phase 0–1 planning (done here) ~10m; definitions ~5m; proofs ~20m; verification code +
runs ~15m; report ~10m. Buffer ~10m.

## Potential Challenges
- Asymmetric-prior gap (Thm 1 caveat) → restrict clean N&S to symmetric/uniform prior.
- Sufficient≠necessary (Van den Berg guardrail) → prove necessity via explicit lower bound.
- Exact vs local Hellinger contraction → state exact χ² result + verify h² numerically.

## Success Criteria
Complete, gap-free proofs of Lemmas 2–5 and Theorems 6–8, each with a passing
computational check, and a REPORT.md stating the results and the diagnostic rule.
