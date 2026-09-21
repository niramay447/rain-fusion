"""
Diagnostic for a low predictive-interval coverage number.

Run this on the per-node MC-dropout CSV produced by mc_dropout_uncertainty.py
(columns: target, pred_mean, pred_std, pred_sigma_total, pred_lo, pred_hi,
within_interval, within_interval_predictive).

It prints the handful of numbers needed to tell WHY coverage is low. The key
invariant: sqrt(tau^-1) should be close to the model's test RMSE. If
sigma_obs << RMSE, the observation-noise term was calibrated on data the model
fits better than the test set (or was computed wrongly), and the intervals are
too narrow by construction.

Usage:
    python diagnose_coverage.py path/to/mc_dropout_f0.csv
"""
import sys

import numpy as np
import pandas as pd


def main(path):
    d = pd.read_csv(path)
    need = ['target', 'pred_mean', 'pred_std', 'pred_sigma_total',
            'pred_lo', 'pred_hi', 'within_interval_predictive']
    missing = [c for c in need if c not in d.columns]
    if missing:
        print(f"MISSING COLUMNS: {missing}")
        print(f"present: {list(d.columns)}")
        return

    t = d['target'].values
    mu = d['pred_mean'].values
    s_mc = d['pred_std'].values
    s_tot = d['pred_sigma_total'].values
    dry = t == 0
    rmse = float(np.sqrt(np.mean((t - mu) ** 2)))
    # sigma_total^2 = sigma_mc^2 + tau^-1  ->  recover tau^-1
    tau_inv = float(np.median(s_tot ** 2 - s_mc ** 2))
    sigma_obs = float(np.sqrt(max(tau_inv, 0.0)))

    print(f"n = {len(d)}   dry fraction = {dry.mean():.4f}")
    print()
    print("--- THE KEY CHECK -------------------------------------------")
    print(f"  test RMSE (from this file)   = {rmse:.4f}")
    print(f"  sigma_obs = sqrt(tau^-1)     = {sigma_obs:.4f}")
    print(f"  ratio sigma_obs / RMSE       = {sigma_obs / rmse:.3f}   <-- want ~0.9-1.1")
    if sigma_obs < 0.5 * rmse:
        print("  >> FAIL: tau^-1 is far too small for the test residuals.")
        print("     Either it was estimated on a split the model fits much better,")
        print("     or it was computed as a std instead of a variance, or in")
        print("     normalised rather than physical units.")
    elif sigma_obs > 2.0 * rmse:
        print("  >> tau^-1 much larger than test error; intervals will over-cover.")
    else:
        print("  >> OK: tau^-1 is on the right scale.")
    print()
    print("--- variance decomposition ----------------------------------")
    print(f"  mean sigma_MC (dropout only) = {s_mc.mean():.4f}")
    print(f"  mean sigma_total             = {s_tot.mean():.4f}")
    frac = 1 - (s_mc.mean() ** 2) / (s_tot.mean() ** 2) if s_tot.mean() > 0 else float('nan')
    print(f"  tau term = {100*frac:.1f}% of total variance   <-- want >90%")
    if frac < 0.5:
        print("  >> FAIL: tau^-1 is not dominating. It was probably not added,")
        print("     or added to the wrong quantity.")
    print()
    print("--- interval sanity -----------------------------------------")
    print(f"  median interval [lo, hi]     = [{np.median(d['pred_lo']):.3f}, {np.median(d['pred_hi']):.3f}]")
    print(f"  median width                 = {np.median(d['pred_hi'] - d['pred_lo']):.3f}")
    print(f"  frac with pred_lo == 0       = {(d['pred_lo'] == 0).mean():.4f}   <-- want high if targets are zero-inflated")
    print()
    print("--- coverage breakdown --------------------------------------")
    print(f"  coverage (all)               = {d['within_interval_predictive'].mean():.4f}")
    print(f"  coverage | dry (target==0)   = {d['within_interval_predictive'][dry].mean():.4f}   <-- want ~0.99")
    print(f"  coverage | wet (target>0)    = {d['within_interval_predictive'][~dry].mean():.4f}   <-- want ~0.85")
    if 'within_interval' in d.columns:
        print(f"  (epistemic-only, for reference) = {d['within_interval'].mean():.4f}")
    print()
    print("--- target sanity -------------------------------------------")
    print(f"  target: min={t.min():.3f} max={t.max():.3f} mean={t.mean():.4f}")
    print(f"  pred  : min={mu.min():.3f} max={mu.max():.3f} mean={mu.mean():.4f}")
    if mu.min() < -1e-9:
        print("  >> predictions are negative — output not clamped at 0?")
    if abs(t.mean()) > 1e-9 and abs(mu.mean() / t.mean() - 1) > 0.5:
        print("  >> prediction and target means differ by >50% — unit or scale mismatch?")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
