#!/usr/bin/env python3
"""
Diagnose whether the Wagga CRRPH satellite data is usable, or whether it is the
near-uniform "everything is dry" prior that collapses gauge+radar+satellite.

Runs against the BUILT satellite cache pkl (rows: timestamp, data=(n_nodes,2)
[accum_mm, valid_flag]) — no netCDF reads needed. Node coords are reconstructed
from the preprocessor geometry. The gauge cross-correlation (the decisive test)
additionally needs the local gauge CSVs.

Usage (on Gadi, from repo root):
    python3 inspect_satellite_wagga.py \
        --sat database/australia/satellite_wagga_2022.pkl \
        --gauge-long database/australia/wagga_rainfall_long.csv \
        --gauge-meta database/australia/wagga_station_metadata.csv

Read the VERDICT block at the end.
"""
import argparse
import pickle

import numpy as np
import pandas as pd

WET = 0.5  # mm/h threshold for "wet" (matches the F1 threshold used in training)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def infer_stride(n_nodes):
    """Find the preprocessor stride whose node count matches the cache."""
    from src.satellite.wagga_satellite_preprocessor import AUSatellitePreprocessor
    for s in range(1, 12):
        if AUSatellitePreprocessor(stride=s).n_nodes == n_nodes:
            return s, AUSatellitePreprocessor(stride=s)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", required=True)
    ap.add_argument("--gauge-long", default=None)
    ap.add_argument("--gauge-meta", default=None)
    ap.add_argument("--min-lag", type=int, default=-6,
                    help="min satellite-vs-gauge shift in hours")
    ap.add_argument("--max-lag", type=int, default=24,
                    help="max shift; +10/+11h would reveal a UTC↔AEST tz bug")
    args = ap.parse_args()

    with open(args.sat, "rb") as f:
        df = pickle.load(f)
    print(f"Loaded {args.sat}: {len(df)} timesteps")

    data = np.stack(df["data"].to_numpy())  # (T, n_nodes, 2)
    T, n_nodes, _ = data.shape
    accum = data[:, :, 0]   # (T, n_nodes) mm
    valid = data[:, :, 1]   # (T, n_nodes) 1/0
    print(f"T={T}  n_nodes={n_nodes}")

    stride, prep = infer_stride(n_nodes)
    print(f"inferred stride = {stride}")

    # ── A. validity ───────────────────────────────────────────────────────────
    val_rate = valid.mean()
    per_step_valid = valid.mean(axis=1)
    all_invalid_steps = int((per_step_valid == 0).sum())
    print("\n── A. RETRIEVAL VALIDITY ──")
    print(f"overall valid fraction        : {val_rate:.3f}")
    print(f"timesteps fully invalid (all 0): {all_invalid_steps} / {T} "
          f"({all_invalid_steps / T:.1%})")
    print(f"valid-fraction percentiles 10/50/90: "
          f"{np.percentile(per_step_valid, [10, 50, 90]).round(3)}")

    # ── B. wet fraction & magnitude (over VALID pixels only) ───────────────────
    vmask = valid > 0.5
    acc_valid = accum[vmask]
    wet_frac = (acc_valid >= WET).mean() if acc_valid.size else float("nan")
    print("\n── B. RAIN SIGNAL (valid pixels only) ──")
    print(f"valid pixel-hours             : {acc_valid.size:,}")
    print(f"wet fraction (>= {WET} mm/h)    : {wet_frac:.4f}")
    if acc_valid.size:
        nz = acc_valid[acc_valid > 0]
        print(f"nonzero accum percentiles 50/90/99/max: "
              f"{np.percentile(nz, [50, 90, 99]).round(2) if nz.size else 'n/a'}  "
              f"max={acc_valid.max():.2f}")
        print(f"domain-mean rain over time    : mean={accum.mean():.4f} "
              f"max-step-mean={accum.mean(axis=1).max():.3f}")

    # ── C. DECISIVE: gauge <-> nearest-sat-pixel correlation ───────────────────
    if not (args.gauge_long and args.gauge_meta and prep is not None):
        print("\n(skipping gauge correlation — pass --gauge-long/--gauge-meta)")
        return

    print("\n── C. GAUGE ↔ SATELLITE CORRELATION (decisive) ──")
    meta = pd.read_csv(
        args.gauge_meta, header=None,
        names=["name", "station_id", "_x", "latitude", "longitude"],
    )
    meta["station_id"] = meta["station_id"].astype(str).str.zfill(6)
    g = pd.read_csv(args.gauge_long, dtype={"station_id": str})
    g["station_id"] = g["station_id"].str.zfill(6)
    # gauge timestamps are tz-naive Australia/Sydney → convert to naive-UTC clock
    # to match sat_ts, exactly as satellitegraph_au.py does in training.
    g["timestamp"] = (
        pd.to_datetime(g["timestamp"])
        .dt.tz_localize("Australia/Sydney", ambiguous="NaT",
                        nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    g = g.dropna(subset=["timestamp"])

    # Replicate the TRAINING pipeline's tz handling (satellitegraph_au.py):
    # gauge timestamps are tz-naive LOCAL (Australia/Sydney); satellite is UTC.
    # Convert both to a common tz-naive-UTC clock so lag 0 == pipeline alignment.
    sat_ts = (
        pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
        if pd.to_datetime(df["timestamp"]).dt.tz is not None
        else pd.to_datetime(df["timestamp"])
    ).to_numpy()
    slat, slon = prep.lat, prep.lon

    LAGS = range(args.min_lag, args.max_lag + 1)  # hours to shift sat vs gauge
    rs, n_used = [], 0
    lag_r = {lag: [] for lag in LAGS}  # median r per lag (rules out misalignment)
    for _, row in meta.iterrows():
        sid = row["station_id"]
        gi = g[g["station_id"] == sid]
        if gi.empty:
            continue
        # nearest satellite node
        d = haversine_km(row["latitude"], row["longitude"], slat, slon)
        node = int(np.argmin(d))
        sat_acc = pd.Series(accum[:, node], index=sat_ts)
        sat_val = pd.Series(valid[:, node] > 0.5, index=sat_ts)
        gser = gi.set_index("timestamp")["rainfall_mm"]
        # drop duplicate timestamps (DST fall-back repeats a local hour; raw
        # feeds can also dup) — non-unique indices break reindex/alignment.
        gser = gser[~gser.index.duplicated(keep="first")].sort_index()
        sat_acc = sat_acc[~sat_acc.index.duplicated(keep="first")].sort_index()
        sat_val = sat_val[~sat_val.index.duplicated(keep="first")].sort_index()
        for lag in LAGS:
            # shift satellite by `lag` hours, then align on the common index
            shifted = sat_acc.index + np.timedelta64(lag, "h")
            sat_series = pd.Series(sat_acc.to_numpy(), index=shifted)
            sv = pd.Series(sat_val.to_numpy(), index=shifted)
            common = gser.index.intersection(sat_series.index)
            if len(common) < 100:
                continue
            gg = gser.reindex(common)
            ss = sat_series.reindex(common)
            vv = sv.reindex(common).fillna(False).to_numpy().astype(bool)
            gg, ss = gg[vv], ss[vv]
            if len(gg) < 100 or gg.std() == 0 or ss.std() == 0:
                continue
            r = np.corrcoef(gg, ss)[0, 1]
            lag_r[lag].append(r)
            if lag == 0:
                rs.append(r)
                n_used += 1
    rs = np.array(rs)
    print(f"gauges matched & usable       : {n_used}")
    if rs.size:
        print(f"gauge-vs-sat Pearson r (lag 0)  median={np.median(rs):.3f}  "
              f"mean={rs.mean():.3f}  min={rs.min():.3f}  max={rs.max():.3f}")
    print("median r by lag (rules out temporal misalignment):")
    for lag in LAGS:
        v = lag_r[lag]
        print(f"   lag {lag:+d}h : median r = "
              f"{np.median(v):.3f}" if v else f"   lag {lag:+d}h : n/a")
    best_lag = max(LAGS, key=lambda L: np.median(lag_r[L]) if lag_r[L] else -1)
    best = np.median(lag_r[best_lag]) if lag_r[best_lag] else float("nan")
    print(f"BEST lag = {best_lag:+d}h  (median r = {best:.3f})")

    # ── VERDICT ────────────────────────────────────────────────────────────────
    print("\n══════════════ VERDICT ══════════════")
    med_r = float(np.median(rs)) if rs.size else float("nan")
    print(f"valid fraction = {val_rate:.2f} | wet fraction = {wet_frac:.4f} | "
          f"median gauge-sat r = {med_r:.3f}")
    print("Interpretation:")
    print("  median r >~ 0.3  → satellite carries real signal; collapse is a")
    print("                     WIRING problem (gate on valid flag / normalize /")
    print("                     loss-weight) — worth fixing, keep satellite.")
    print("  median r ~ 0     → satellite is signal-free over these gauges; no")
    print("                     architecture fix helps — drop it (honest finding).")
    print("  valid<<1 or wet~0 → mostly a dry/invalid prior → confirms the")
    print("                     collapse mechanism; try valid-flag gating first.")


if __name__ == "__main__":
    main()
