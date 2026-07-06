#!/usr/bin/env python3
"""Download selected arXiv papers to papers/."""
import httpx, os, sys

PAPERS = {
    "2210.10760": "gao2022_scaling_laws_reward_overoptimization",
    "2403.16981": "pensia2024_sample_complexity_binary_hypothesis_testing",
    "1507.00473": "hanneke2015_optimal_sample_complexity_pac_learning",
    "2403.03375": "qiu2024_complexity_matters_feature_learning_spurious",
    "2402.12715": "ye2024_clever_hans_mirage_spurious_survey",
    "2105.04026": "berner2021_modern_mathematics_deep_learning",
    "2012.11854": "zhu2020_instance_dependent_label_noise",
    "2310.15450": "jin2023_identifiability_causal_representation_learning",
    "2501.08887": "vandenberg2025_pac_learnability_necessary_sufficient",
    "2310.02743": "coste2023_reward_model_ensembles_overoptimization",
    "2505.12763": "zhang2025_reward_overoptimization_evaluation",
    "2412.07942": "brill2024_neural_scaling_laws_data_distribution",
    "1806.02419": "rafi2018_likelihood_alternative_nhst",
    "2305.18290": "rafailov2023_direct_preference_optimization",
}

os.makedirs("papers", exist_ok=True)
ok, fail = [], []
for aid, name in PAPERS.items():
    dest = f"papers/{aid}_{name}.pdf"
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        ok.append(dest); continue
    url = f"https://arxiv.org/pdf/{aid}.pdf"
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        if os.path.getsize(dest) > 10000:
            ok.append(dest)
            print(f"OK  {dest} ({len(r.content)//1024} KB)")
        else:
            fail.append(aid); print(f"SMALL {aid}")
    except Exception as e:
        fail.append(aid); print(f"FAIL {aid}: {e}")

print(f"\nDownloaded {len(ok)}/{len(PAPERS)}. Failures: {fail}")
