# Literature Review: Mathematical Characterization of Null Robustness in LLM Alignment via Synthetic Dataset Topology

## Research Area Overview

The project asks a **statistical-learning-theory** question dressed in an
alignment-engineering setting: *when an alignment intervention (RLHF, DPO, reward
modeling, safety fine-tuning) produces no measurable change, is that because the
model is genuinely robust, or because the experimental dataset lacks the
mathematical structure needed to reveal the effect?*

The hypothesis reduces to three coupled mathematical claims:

- **(H1) Detectability = divergence vs. sample size.** An intervention shifting an
  outcome distribution from `q` to `p` is detectable with `n` samples iff a
  statistical divergence `D(p,q)` exceeds a threshold set by `n` and the target
  error `δ`. This is exactly the *sample complexity of simple binary hypothesis
  testing*.
- **(H2) Dataset topology modulates the effective divergence.** Feature-correlation
  structure (spurious correlations), label-noise distributions, and causal-graph
  complexity each *shrink* the achievable divergence between aligned and unaligned
  outputs on a given dataset, pushing experiments below the detectability threshold.
- **(H3) Synthetic datasets can dial these knobs.** Boolean-function constructions
  give parametric control over feature complexity, correlation strength, and noise,
  enabling controlled search for the necessary/sufficient boundary.

The relevant subfields are **mathematical statistics (math.ST)**, **statistical
learning theory (cs.LG / stat.ML)**, **information theory**, and **causal
inference**. Applicable MSC codes: 62G10 (hypothesis testing), 68Q32 (computational
learning theory), 62B10 (information-theoretic statistics), 62H22 (graphical models).

---

## Key Definitions

**Definition 1 (Simple binary hypothesis testing, prior-free).** Given two
distributions `p, q` on a domain `X`, a test `φ` solves `B_PF(p,q,α,β)` with sample
complexity `n` if, on `n` i.i.d. samples, `P_{X∼p^⊗n}[φ≠p] ≤ α` (type-I) and
`P_{X∼q^⊗n}[φ≠q] ≤ β` (type-II). `n*_{PF}(p,q,α,β)` is the least such `n`.
(Pensia–Jog–Loh 2024, Def. 1.2.)

**Definition 2 (Squared Hellinger divergence).**
`h²(p,q) = 1 − Σ_i √(p_i q_i) = ½ Σ_i (√p_i − √q_i)² ∈ [0,1]`. Related to Chernoff
information `C_I(p,q)` by `h²(p,q) ≍ C_I(p,q)` under mild conditions.

**Definition 3 (Jensen–Shannon divergence).**
`JS(p,q) = ½ KL(p‖m) + ½ KL(q‖m)`, `m = ½(p+q)`. Bounded, symmetric; a member of the
Jensen–Shannon divergence family used to characterize sample complexity in the
asymmetric-prior regime.

**Definition 4 (Minimum detectable effect / divergence).** For fixed `n`, `δ`, the
smallest divergence an experiment can resolve. Under the uniform-prior
characterization, `h²_min(n,δ) ≍ log(1/δ)/n`.

**Definition 5 (Boolean spurious-feature dataset).** (Qiu et al. 2024, §2.) Fix a
*core* feature `f_c:{±1}^c→{±1}` (fully predicts the label `y=f_c(x_c)`) and a
*spurious* feature `f_s:{±1}^s→{±1}` of lower complexity. Draw `x ~ D_λ` where
`D_λ = λ·D_same + (1−λ)·D_diff`, `D_same` upsampling `{f_s=f_c}`. Then `λ∈[½,1]` is
the **correlation strength**, `c,s` are **complexities**, and
`x_s — f_s — y — f_c — x_c` forms a **Markov chain** (the causal graph).

**Definition 6 (Instance-dependent label noise).** Noisy label `ỹ` with flip rate
`T(x) = P(ỹ≠y | x)` depending on the instance `x`, not only the class `y`
(Zhu–Song–Liu 2020). Contrast class-dependent noise `T` constant within a class.

**Definition 7 (Reward overoptimization / Goodhart gap).** With gold reward `R_gold`
and proxy `R_proxy`, optimizing `R_proxy` past a point *decreases* `R_gold`. Measured
against `d := √(KL(π‖π_init))` (Gao et al. 2022).

---

## Known Results (Prerequisite Theorems)

**Theorem 1 (Sample complexity of hypothesis testing — Pensia–Jog–Loh 2024).**
In the Bayesian uniform-prior regime, under `h²(p,q) ≤ ½`, `δ ≤ ¼`,
> `n*_B(p,q,½,δ) ≍ log(1/δ) / h²(p,q)`.
The paper extends the tight characterization to all `0 ≤ α,β ≤ 1/8` (prior-free) and
all `δ ≤ π/4` (Bayesian), via equivalent expressions in the Jensen–Shannon and
Hellinger divergence families. **Caveat (asymmetry):** for skewed priors the
complexity can be asymmetric in `p,q` and is *not* captured by Hellinger alone
(their Bernoulli(0) vs Bernoulli(ε) example).
- *Used for:* the master detectability inequality (H1). An alignment effect of
  "size" `h²` is detectable iff `n ≳ log(1/δ)/h²`.

**Theorem 2 (Optimal PAC sample complexity — Hanneke 2015).** For a hypothesis class
of VC dimension `d`, realizable PAC learning to error `ε` with confidence `1−δ` needs
> `n = Θ( (d + log(1/δ)) / ε )` samples,
matching the lower bound up to constants.
- *Used for:* the learnability-threshold analogue; separates "an effect exists but is
  unlearnable at this `n`" from "no effect."

**Theorem 3 (Reward-overoptimization functional forms — Gao et al. 2022).**
With `d = √(KL(π‖π_init))`, the gold-reward frontier follows
> Best-of-n:  `R_bon(d) = d(α_bon − β_bon d)`
> RL (PPO):   `R_RL(d)  = d(α_RL − β_RL log d)`,
with `R(0)=0`. Coefficients `α,β` scale **smoothly (≈ logarithmically)** with proxy-RM
parameter count and dataset size. **Empirical threshold:** below ≈2,000 comparisons,
RMs show "very little improvement over near-chance loss" — a null-result regime driven
by dataset size, not model robustness.
- *Used for:* the alignment-specific instantiation of H1/H2 and the synthetic gold-RM
  experimental design the project generalizes.

**Theorem 4 (DPO reparameterization — Rafailov et al. 2023).** The RLHF optimum under
a KL-regularized objective has closed form `π*(y|x) ∝ π_ref(y|x) exp(R(x,y)/β)`, so the
reward is recoverable as `R(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)`; preference
learning reduces to a binary classification loss on `(π/π_ref)` log-ratios.
- *Used for:* expressing the alignment intervention as a distribution shift whose
  divergence from baseline is the quantity in Theorem 1.

**Theorem 5 (Second-order label-noise correction — Zhu–Song–Liu 2020).** Under
instance-dependent noise, first-order (class-conditional) corrections are biased;
covariance terms `Cov(noise rate, Bayes-optimal label)` capture the induced imbalance
and restore consistency of the peer-loss estimator.
- *Used for:* the label-noise axis (H2) — noise flattens the class-conditional
  distributions, shrinking `h²(p,q)` and inflating the detectability threshold.

**Theorem 6 (Causal identifiability — Jin et al. 2023).** In a general nonparametric
latent causal model with a general transformation to observations, **two hard
uncoupled interventions per node** suffice for perfect recovery of the latent causal
graph and variables (identifiability + achievability).
- *Used for:* the causal-graph-complexity axis (H2): more nodes/edges ⇒ more
  interventions (samples) needed before causal signal becomes identifiable/detectable.

**Theorem 7 (Sufficient ≠ necessary for PAC — Van den Berg et al. 2025).** For
scenario decision-making, finite VC dimension and existence of a compression scheme
are **sufficient but not necessary** for the PAC property — unlike binary
classification, where they are both.
- *Used for:* a rigor guardrail. Any "necessary and sufficient conditions" claim in
  this project must be proven within a fixed setting; conditions that are N&S for
  detectability under symmetric priors may fail to be necessary under skew (see
  Theorem 1 caveat).

**Result 8 (Scaling laws from data topology — Brill et al. 2024).** Modeling natural
datasets via percolation theory yields two criticality regimes, each producing
power-law scaling exponents determined by the data distribution's structure.
- *Used for:* precedent that *dataset topology* (not model) sets the achievable
  error/effect curve.

---

## Proof Techniques in the Literature

- **f-divergence inequalities / information theory.** Pensia et al. reduce sample
  complexity to a novel `f`-divergence inequality between Jensen–Shannon and Hellinger
  families, proved by error/success amplification reductions plus case analysis. This
  is the primary technique for our detectability theorems.
- **Boolean function analysis (Fourier/parity, staircase functions).** Qiu et al. use
  parity `χ_S(x)=Π_{i∈S} x_i` and *leap-1 degree-d staircase* functions to set feature
  complexity exactly; learning dynamics analyzed via degree profiles. This is the
  cleanest way to give a synthetic dataset a *tunable* correlation topology.
- **Chernoff–Stein / large-deviations asymptotics.** Give `e^{−n(C_I+o(1))}` error
  rates; use for asymptotic sanity checks but *not* for finite-sample thresholds
  (the paper's central warning).
- **PAC / VC & compression-scheme arguments.** Hanneke's optimal bound and the
  scenario-decision counterexamples; use for learnability thresholds and for checking
  necessity claims.
- **Score-based identification.** Jin et al. estimate the transformation inverse from
  score variations across intervention environments — the causal-graph technique.
- **Empirical functional-form fitting + advance prediction.** Gao et al. hypothesize a
  form on low-KL data and validate by extrapolation; a template for how to *state and
  test* a conjectured detectability curve.

---

## Related Open Problems

- **Asymmetric-prior detectability.** Pensia et al. resolve `α=β` / `π=½`; a full
  characterization for all priors *and* all error regimes with a single closed-form
  divergence remains open. Directly relevant since alignment "success" events are rare
  (skewed prior).
- **Proxy-reward score modeling.** Gao et al. could not fit the *proxy* (as opposed to
  gold) reward curve — the overoptimization gap lacks a clean functional form.
- **Necessary conditions for PAC beyond classification.** Van den Berg et al. leave
  open what *is* necessary for the PAC property in scenario decision-making.
- **Instance-dependent noise minimax rates.** General learnability rates under
  arbitrary `T(x)` are not pinned down.

---

## Gaps and Opportunities

- **No unified "null-robustness" theorem.** The literature has (a) detectability
  thresholds and (b) mechanisms that shrink divergence (spurious correlation, noise,
  causal complexity) — but *no result ties dataset structural parameters `(c,s,λ,T,
  graph)` to the effective `h²` and hence to a detectability threshold.* This is the
  project's central opportunity.
- **Divergence contraction bounds.** Missing: a quantitative bound of the form
  `h²_effective ≤ g(λ, noise rate, graph complexity) · h²_ideal`. A data-processing /
  strong-data-processing-inequality (SDPI) argument is the natural tool.
- **Synthetic ⇄ alignment bridge.** Gao et al.'s gold-RM setup and Qiu et al.'s
  Boolean datasets have never been combined into one parametric testbed.

---

## Recommendations for Proof Strategy

- **Recommended approach.** Build the master result on **Theorem 1** (detectability
  threshold `n ≳ log(1/δ)/h²`). Then prove **divergence-contraction lemmas** showing
  each dataset property multiplies `h²` by an explicit factor:
  - *Spurious correlation:* via the Markov chain `x_s—f_s—y—f_c—x_c` and a
    **data-processing inequality**, bound how correlation `λ` caps the mutual
    information / divergence carried by the core signal.
  - *Label noise:* a noisy channel `T(x)` is a stochastic post-processing; SDPI gives
    `h²(T p, T q) ≤ η · h²(p,q)` with contraction coefficient `η<1` — quantify `η(T)`.
  - *Causal complexity:* use Theorem 6's "interventions-per-node" count as the sample
    multiplier; graph complexity ⇒ larger effective `n` needed.
- **Key lemmas to establish.**
  1. Detectability inequality (cite Theorem 1) restated for alignment shifts.
  2. Contraction lemma per axis (correlation, noise, causal), each yielding an explicit
     `h²_eff`.
  3. Composition: combine contractions ⇒ a single sufficient condition
     `n ≥ log(1/δ) / (η_corr·η_noise·η_causal · h²_ideal)` and a matching necessary
     condition (lower bound) to claim N&S.
- **Potential obstacles.**
  - The **asymmetric-prior gap** (Theorem 1 caveat): rare-success alignment events may
    break the clean Hellinger characterization — restrict to symmetric/uniform-prior
    detectability first.
  - **Sufficient ≠ necessary** (Theorem 7): prove necessity via explicit lower-bound
    constructions, don't infer it.
  - Boolean-analysis dynamics (Qiu et al.) show spurious features are "not forgotten"
    even after core learning — the *static* divergence picture may need a *dynamic*
    (training-trajectory) refinement.
- **Computational support.** Use `code/verification/detectability.py`: numerically
  compute `h², JS, TV`, the sample-complexity threshold, and generate the parametric
  Boolean dataset to empirically trace the null→detectable boundary and validate the
  contraction lemmas before/after committing to constants.
