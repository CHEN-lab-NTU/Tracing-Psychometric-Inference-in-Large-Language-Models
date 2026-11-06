#!/usr/bin/env python3
"""Human-matrix bootstrap from the original study robustness check.

This is check_bootstrap_human from LLM_Psychometric/mech/robustness_checks.py.
Only package imports and this standalone entry point differ.

Usage:
    PYTHONPATH=src python scripts/analysis/compute_human_bootstrap.py
"""

import numpy as np
from scipy.stats import pearsonr


def check_bootstrap_human():
    print("="*60)
    print("  CHECK 3: Bootstrap CI on human correlation matrix")
    print("="*60)

    from psychometric_inference.mechanisms.config import ALL_SUBSCALES
    from psychometric_inference.mechanisms.cross_persona import load_human_scores

    scores_df = load_human_scores()
    available = [s for s in ALL_SUBSCALES if s in scores_df.columns]
    n_sub = len(available)
    n_subjects = len(scores_df)

    # Real correlation matrix
    real_corr = scores_df[available].corr()
    idx = np.triu_indices(n_sub, k=1)
    real_vec = real_corr.values[idx]

    # Bootstrap
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_matrices = []

    for _ in range(n_boot):
        boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_df = scores_df.iloc[boot_idx]
        boot_corr = boot_df[available].corr().values[idx]
        boot_matrices.append(boot_corr)

    boot_matrices = np.array(boot_matrices)  # (n_boot, n_pairs)

    # CI per pair
    ci_lower = np.percentile(boot_matrices, 2.5, axis=0)
    ci_upper = np.percentile(boot_matrices, 97.5, axis=0)
    ci_width = ci_upper - ci_lower

    print(f"\n  {n_sub} subscales, {len(real_vec)} pairs, {n_subjects} subjects")
    print(f"  Bootstrap: {n_boot} iterations")
    print(f"\n  95% CI width statistics:")
    print(f"    Mean CI width:   {ci_width.mean():.3f}")
    print(f"    Median CI width: {np.median(ci_width):.3f}")
    print(f"    Max CI width:    {ci_width.max():.3f}")
    print(f"    Min CI width:    {ci_width.min():.3f}")

    # Which pairs have widest CIs?
    pair_labels = []
    for i, j in zip(*idx):
        pair_labels.append(f"{available[i].split('_',1)[1][:10]}-{available[j].split('_',1)[1][:10]}")

    wide_idx = np.argsort(ci_width)[-5:]
    print(f"\n  Widest CI pairs:")
    for wi in wide_idx:
        print(f"    {pair_labels[wi]:<30} r = {real_vec[wi]:.3f} [{ci_lower[wi]:.3f}, {ci_upper[wi]:.3f}] (width = {ci_width[wi]:.3f})")

    # Overall stability: bootstrap correlation of the full vector
    boot_mantel_rs = []
    for bm in boot_matrices:
        r, _ = pearsonr(real_vec, bm)
        boot_mantel_rs.append(r)
    boot_mantel_rs = np.array(boot_mantel_rs)

    print(f"\n  Stability of full correlation structure:")
    print(f"    Pearson r (real vs bootstrap): mean = {boot_mantel_rs.mean():.4f}, "
          f"95% CI = [{np.percentile(boot_mantel_rs, 2.5):.4f}, {np.percentile(boot_mantel_rs, 97.5):.4f}]")
    print()


if __name__ == "__main__":
    check_bootstrap_human()
