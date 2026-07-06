# Downloaded Papers

Curated for: *Mathematical Characterization of Null Robustness in LLM Alignment
via Synthetic Dataset Topology*. Papers are grouped by the mathematical theme
they support. All are arXiv PDFs; filename = `{arxiv_id}_{author_year_topic}.pdf`.

## A. Detectability & statistical power (the "necessary/sufficient condition" core)

1. **The Sample Complexity of Simple Binary Hypothesis Testing** — `2403.16981_pensia2024_sample_complexity_binary_hypothesis_testing.pdf`
   - Authors: Pensia, Jog, Loh · Year: 2024 (COLT 2024; math.ST)
   - Why relevant: **Foundational.** Gives the exact characterization of how many
     i.i.d. samples are needed to distinguish two distributions p, q:
     `n* ≍ log(1/δ) / h²(p,q)` (Hellinger divergence). This is the mathematical
     backbone for "when does an intervention produce a *detectable* effect."

2. **The Optimal Sample Complexity of PAC Learning** — `1507.00473_hanneke2015_optimal_sample_complexity_pac_learning.pdf`
   - Authors: Hanneke · Year: 2015 (JMLR)
   - Why relevant: Tight realizable-case PAC bound `n = Θ((d + log(1/δ))/ε)`;
     canonical learnability threshold that separates "learnable" from "artifact."

3. **A Likelihood-based Alternative to Null Hypothesis Significance Testing** — `1806.02419_rafi2018_likelihood_alternative_nhst.pdf`
   - Why relevant: Frames what a "null result" *means* statistically; supports the
     claim that failure-to-reject ≠ evidence of robustness.

## B. Reward overoptimization / Goodhart (alignment-specific detectability)

4. **Scaling Laws for Reward Model Overoptimization** — `2210.10760_gao2022_scaling_laws_reward_overoptimization.pdf`
   - Authors: Gao, Schulman, Hilton · Year: 2022 · **CENTRAL PAPER**
   - Why relevant: Introduces the *synthetic gold-RM* methodology this project
     generalizes, and the functional forms for the gold-reward frontier. Reports a
     **data threshold (~2,000 comparisons)** below which interventions show "near
     chance" effects — a concrete null-result-from-dataset-structure phenomenon.

5. **Reward Model Ensembles Help Mitigate Overoptimization** — `2310.02743_coste2023_reward_model_ensembles_overoptimization.pdf`
   - Authors: Coste et al. · Year: 2023
   - Why relevant: Extends (4); shows when overoptimization effects appear/vanish.

6. **Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization** — `2505.12763_zhang2025_reward_overoptimization_evaluation.pdf`
   - Year: 2025 · Why relevant: recent evaluation-side view of detectability.

7. **Direct Preference Optimization: Your LM is Secretly a Reward Model** — `2305.18290_rafailov2023_direct_preference_optimization.pdf`
   - Authors: Rafailov et al. · Year: 2023
   - Why relevant: Closed-form policy/reward link; defines the alignment
     intervention whose effect we test for detectability.

## C. Dataset structure: spurious correlations & feature topology

8. **Complexity Matters: Dynamics of Feature Learning in the Presence of Spurious Correlations** — `2403.03375_qiu2024_complexity_matters_feature_learning_spurious.pdf`
   - Authors: Qiu et al. · Year: 2024 · **Methodological template**
   - Why relevant: Constructs a synthetic Boolean-function dataset with
     *parametric control* of core complexity `c`, spurious complexity `s`, and
     correlation strength `λ` — exactly the "parametrically controlled mathematical
     properties" the hypothesis calls for. Includes a Markov-chain causal structure.

9. **The Clever Hans Mirage: A Comprehensive Survey on Spurious Correlations** — `2402.12715_ye2024_clever_hans_mirage_spurious_survey.pdf`
   - Year: 2024 (survey) · Why relevant: taxonomy of spurious-correlation settings,
     datasets, metrics; maps the landscape of "feature correlation topology."

## D. Label noise (label-noise distribution axis of the hypothesis)

10. **A Second-Order Approach to Learning with Instance-Dependent Label Noise** — `2012.11854_zhu2020_instance_dependent_label_noise.pdf`
    - Authors: Zhu, Song, Liu · Year: 2020
    - Why relevant: Instance-dependent noise ≠ class-dependent noise; second-order
      (covariance) statistics govern learnability under noise.

## E. Causal graph complexity

11. **General Identifiability and Achievability for Causal Representation Learning** — `2310.15450_jin2023_identifiability_causal_representation_learning.pdf`
    - Year: 2023 · Why relevant: identifiability conditions (interventions per node)
      as a function of causal-graph structure — the "causal graph complexity" axis.

12. **PAC Learnability of Scenario Decision-Making: Necessary vs. Sufficient Conditions** — `2501.08887_vandenberg2025_pac_learnability_necessary_sufficient.pdf`
    - Year: 2025 · Why relevant: shows finite-VC / compression are **sufficient but
      not necessary** for PAC — a caution for "necessary and sufficient conditions"
      claims that this project must respect.

## F. Scaling laws from data distribution

13. **Neural Scaling Laws Rooted in the Data Distribution** — `2412.07942_brill2024_neural_scaling_laws_data_distribution.pdf`
    - Year: 2024 · Why relevant: percolation-theory model deriving scaling-law
      exponents from *dataset topology* — direct precedent for "dataset topology
      determines detectable effects."

## G. Foundations

14. **The Modern Mathematics of Deep Learning** — `2105.04026_berner2021_modern_mathematics_deep_learning.pdf`
    - Authors: Berner, Grohs, Kutyniok, Petersen · Year: 2021 (survey)
    - Why relevant: reference for approximation/optimization/generalization theory
      used throughout.
