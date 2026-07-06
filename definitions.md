# Definitions and Notation

All distributions are on a finite or Polish outcome space `X`. `p, q` denote
probability measures; `Bern(a)` is the Bernoulli law with `P(1)=a`.

**D1 (Squared Hellinger divergence).**
`h²(p,q) = 1 − ∑_x √(p(x) q(x)) = ½ ∑_x (√p(x) − √q(x))² ∈ [0,1]`.
It is the f-divergence with generator `f_H(t) = ½(√t − 1)²`, which is convex with
`f_H(1)=0`. `h²=0 ⇔ p=q`; `h²=1 ⇔ p⊥q`.

**D2 (χ²-divergence).** `χ²(p‖q) = ∑_x (p(x)−q(x))²/q(x)`. f-divergence with
`f(t)=(t−1)²`.

**D3 (Detectability / sample complexity).** For a simple binary test of `p` vs `q`
on `n` i.i.d. samples at Bayes error `δ`, `n*(p,q;δ)` is the least `n` admitting a
test with error `≤ δ`. (Pensia–Jog–Loh 2024, Def. 1.2; symmetric/uniform prior.)

**D4 (Minimum detectable divergence / floor).** `h²_floor(n,δ) := log(1/δ)/n` — the
smallest `h²` an experiment at `(n,δ)` can resolve (from Theorem 1). An effect with
`h²(p,q) < h²_floor` is *below threshold* and produces a NULL result.

**D5 (Markov kernel / channel).** `K : X → Y` a stochastic map; `Kp` the pushforward
of `p`. Models any post-processing: label corruption, measuring a correlated proxy,
observation through a causal chain.

**D6 (SDPI contraction coefficient).** For a channel `K` and an f-divergence `D_f`,
`η_f(K) := sup_{p≠q} D_f(Kp‖Kq) / D_f(p‖q) ∈ [0,1]`. `η_f(K) ≤ η_{χ²}(K)` for all
operationally-relevant `f` including Hellinger (Cohen–Kemperman–Zbaganu; Raginsky
2016). `η_{TV}(K)` is the Dobrushin coefficient. `η=1` iff `K` preserves some
divergence (informationless-lossless); `η<1` for genuinely noisy channels.

**D7 (Binary symmetric channel).** `BSC(e)`, `e∈[0,½]`, flips a `{±1}` bit with
probability `e`. It maps `Bern(a) ↦ Bern(e + (1−2e)a)`.

**D8 (Alignment intervention as a distribution shift).** An intervention turns the
baseline outcome law `q` into `p`. Its *ideal* strength is `h²_ideal := h²(p,q)`
measured on the target (core) variable. What an experiment observes is
`h²_eff := h²(Kp, Kq)` where `K` is the composed observation channel induced by the
dataset topology.

**D9 (Boolean spurious-feature dataset).** (Qiu et al. 2024.) Core feature
`f_c:{±1}^c→{±1}` with label `y=f_c(x_c)`; spurious feature `f_s:{±1}^s→{±1}` with
`P(f_s=y)=λ∈[½,1]`. Markov chain `x_s — f_s — y — f_c — x_c`. `λ` = **correlation
strength**, `c,s` = **complexities**.

**D10 (Instance-dependent label noise).** Channel with flip rate `T(x)=P(ỹ≠y|x)`;
the *class-symmetric* case is `BSC(e)` with `e = E[T]`. (Zhu–Song–Liu 2020.)

**D11 (Causal path length k).** Length of the latent Markov chain
`Z₁→Z₂→…→Z_k` separating the intervened node from the observed node; each link is a
channel with contraction `≤ η₀ < 1`. (Motivated by Jin et al. 2023.)

**D12 (Null result / genuine robustness).** A **null** is a non-rejection of "no
effect" at `(n,δ)`. It is an **artifact** if the dataset's contraction budget could
push a substantive `h²_ideal` below `h²_floor`; it certifies **genuine robustness**
only for effects with `h²_eff ≥ h²_floor` that still fail to appear.
