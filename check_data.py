#!/usr/bin/env python3
"""
check_data.py
=============
Data verification for the rainfall-fusion project.

Three modes:

  compare  — Compare API-downloaded raw CSV vs a manually-downloaded BoM CSV
             for one station. Shows tip-level differences and hourly aggregation
             comparison.

  verify   — Reconstruct the hourly aggregation from the raw API CSVs and check
             it matches the built long CSV (rainfall_2022_sydney75_long.csv).

  qa       — Timestamp sanity (gaps, DST, coverage) + gauge-gauge spatial
             correlation to confirm no residual timezone offset.

  all      — Run verify + qa together (compare requires --manual-csv).

Usage examples
--------------
  # Compare one station against a manually-downloaded BoM CSV
  python check_data.py compare --station 566008 --manual-csv ~/Downloads/IDCJAC0009_566008_2022.csv

  # Verify the pipeline rebuilt correctly for all stations
  python check_data.py verify --region sydney

  # Timestamp + correlation QA
  python check_data.py qa --region sydney

  # Everything (no manual CSV needed)
  python check_data.py all --region sydney
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DB   = ROOT / "database" / "australia"

REGION_CFG = {
    "sydney": {
        "long_csv":   DB / "rainfall_2022_sydney75_long.csv",
        "meta_csv":   DB / "station_metadata_sydney75.csv",
        "raw_dir":    DB / "raw_sydney",
        "timezone":   "Australia/Sydney",
        "tip_size":   0.5,
        "start":      "2022-01-01",
        "end":        "2022-12-31",
    },
    "wagga": {
        "long_csv":   DB / "wagga_rainfall_long.csv",
        "meta_csv":   DB / "wagga_station_metadata.csv",
        "raw_dir":    DB / "raw_wagga",
        "timezone":   "Australia/Sydney",
        "tip_size":   "auto",
        "start":      "2022-01-01",
        "end":        "2022-12-31",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════

def load_raw_api_csv(path: Path) -> pd.DataFrame:
    """Load a per-station raw API CSV (columns: timestamp, rainfall_mm)."""
    df = pd.read_csv(path, dtype={"rainfall_mm": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0)
    return df


def to_hourly_from_raw(df: pd.DataFrame, timezone: str, tip_size,
                       hourly_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Mirror of build_gauge_dataset.py to_hourly() — reproduce the exact same
    hourly aggregation so we can diff against the built long CSV.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert(timezone).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0)

    mode = str(tip_size).lower()
    if mode not in ("none", "sum", "all"):
        if mode == "auto":
            pos = df.loc[df["rainfall_mm"] > 0, "rainfall_mm"].round(2)
            ts = float(pos.mode().iloc[0]) if len(pos) else None
        else:
            ts = float(tip_size)
        if ts is not None:
            df = df[df["rainfall_mm"].isin([0.0, ts])]

    df = df.set_index("timestamp")
    hourly = df.resample("h").sum()
    hourly = hourly.reindex(hourly_range, fill_value=0)
    return hourly["rainfall_mm"]


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — compare API vs manual BoM CSV
# ════════════════════════════════════════════════════════════════════════════

def _parse_bom_manual(path: Path, station_id: str) -> pd.DataFrame:
    """
    Attempt to parse a manually-downloaded BoM CSV.

    BoM Climate Data Online (IDCJAC0009) format has a header block then columns:
      Product code, Bureau of Meteorology station number, Year, Month, Day,
      Hour, Minute, Rainfall amount (millimetres),
      Period over which rainfall amount measured (minutes), Quality

    We also accept a plain two-column CSV (timestamp, rainfall_mm) in case the
    user exports from the BoM Water Data Online portal (same format as the API).
    """
    # Try to detect format by peeking at first 20 lines
    with open(path) as f:
        head = [f.readline() for _ in range(20)]

    # CDO-style: has "Bureau of Meteorology station number" or "Year,Month,Day"
    if any("Year" in l and "Month" in l and "Day" in l for l in head):
        # Find the actual data header line
        skip = 0
        for i, line in enumerate(head):
            if "Year" in line and "Month" in line:
                skip = i
                break
        df = pd.read_csv(path, skiprows=skip, dtype=str)
        df.columns = [c.strip().strip('"') for c in df.columns]
        # CDO column names may vary; find key columns by substring
        year_col   = next((c for c in df.columns if "Year"   in c), None)
        month_col  = next((c for c in df.columns if "Month"  in c), None)
        day_col    = next((c for c in df.columns if c.strip() == "Day"), None)
        hour_col   = next((c for c in df.columns if "Hour"   in c), None)
        min_col    = next((c for c in df.columns if "Minute" in c), None)
        rain_col   = next((c for c in df.columns if "Rainfall amount" in c or
                           c.strip() == "Rainfall amount (millimetres)"), None)
        if not all([year_col, month_col, day_col, rain_col]):
            print(f"  [!] Cannot identify required columns. Columns found: {list(df.columns)}")
            return pd.DataFrame()
        hour_str  = df[hour_col].fillna("0").str.strip()   if hour_col  else "0"
        min_str   = df[min_col].fillna("0").str.strip()    if min_col   else "0"
        ts_str = (
            df[year_col].str.strip() + "-" +
            df[month_col].str.strip().str.zfill(2) + "-" +
            df[day_col].str.strip().str.zfill(2) + " " +
            hour_str.str.zfill(2) + ":" +
            min_str.str.zfill(2)
        )
        df["timestamp"] = pd.to_datetime(ts_str, errors="coerce")
        df["rainfall_mm"] = pd.to_numeric(
            df[rain_col].str.strip().replace("", np.nan), errors="coerce"
        ).fillna(0)
        df = df.dropna(subset=["timestamp"])
        return df[["timestamp", "rainfall_mm"]].reset_index(drop=True)

    # WaterData / two-column format
    if any("timestamp" in l.lower() for l in head[:3]):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
        df["rainfall_mm"] = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0)
        return df[["timestamp", "rainfall_mm"]].dropna(subset=["timestamp"]).reset_index(drop=True)

    print(f"  [!] Unrecognised format. First 5 lines:")
    for l in head[:5]:
        print(f"      {l.rstrip()}")
    return pd.DataFrame()


def cmd_compare(args):
    cfg = REGION_CFG[args.region]
    sid = str(args.station).zfill(6)
    raw_path = cfg["raw_dir"] / f"{sid}_rainfall_raw.csv"
    manual_path = Path(args.manual_csv)

    if not raw_path.exists():
        sys.exit(f"[!] API raw CSV not found: {raw_path}")
    if not manual_path.exists():
        sys.exit(f"[!] Manual CSV not found: {manual_path}")

    print(f"\n{'='*70}")
    print(f"SECTION 1 — API vs manual CSV   station {sid}")
    print(f"{'='*70}")

    # ── load API raw CSV ─────────────────────────────────────────────────
    api_df = load_raw_api_csv(raw_path)
    api_df["ts_local"] = api_df["timestamp"].dt.tz_convert(cfg["timezone"])
    print(f"\nAPI raw CSV:    {len(api_df):,} rows")
    print(f"  time range:   {api_df['ts_local'].min()}  →  {api_df['ts_local'].max()}")
    print(f"  unique timestamps: {api_df['timestamp'].nunique()}")
    dup_api = api_df['timestamp'].duplicated().sum()
    print(f"  duplicates:   {dup_api}")
    print(f"  value summary (mm):")
    vals = api_df["rainfall_mm"]
    print(f"    count={len(vals)}  mean={vals.mean():.4f}  max={vals.max():.2f}")
    print(f"    value counts (top 5): {dict(vals.value_counts().head(5))}")

    # ── load manual CSV ──────────────────────────────────────────────────
    print(f"\nManual CSV ({manual_path.name}):")
    man_df = _parse_bom_manual(manual_path, sid)
    if man_df.empty:
        print("  [!] Could not parse manual CSV — aborting compare.")
        return
    print(f"  {len(man_df):,} rows")
    print(f"  time range: {man_df['timestamp'].min()}  →  {man_df['timestamp'].max()}")
    man_vals = man_df["rainfall_mm"]
    print(f"  value counts (top 5): {dict(man_vals.value_counts().head(5))}")

    # ── align and compare at tip level ───────────────────────────────────
    print(f"\n── Tip-level comparison ──")

    # Bring API to local tz, strip tz info, to match manual (which may be local)
    api_local = api_df.copy()
    api_local["timestamp"] = api_local["ts_local"].dt.tz_localize(None)
    api_local = api_local.set_index("timestamp")["rainfall_mm"].sort_index()
    api_local = api_local[~api_local.index.duplicated(keep="first")]

    # Manual — detect if tz-naive or tz-aware
    man = man_df.copy()
    if man["timestamp"].dt.tz is not None:
        man["timestamp"] = man["timestamp"].dt.tz_convert(cfg["timezone"]).dt.tz_localize(None)
    man = man.set_index("timestamp")["rainfall_mm"].sort_index()
    man = man[~man.index.duplicated(keep="first")]

    common = api_local.index.intersection(man.index)
    print(f"  Common timestamps: {len(common):,}  (API only: {len(api_local)-len(common):,}  "
          f"manual only: {len(man)-len(common):,})")
    if len(common):
        diff = (api_local.reindex(common) - man.reindex(common)).abs()
        print(f"  Max |diff| at shared timestamps: {diff.max():.4f} mm")
        print(f"  Mean |diff|: {diff.mean():.4f} mm")
        n_mismatch = (diff > 0.001).sum()
        print(f"  Rows with |diff| > 0.001 mm: {n_mismatch} ({n_mismatch/len(common):.1%})")
        if n_mismatch > 0:
            idx = diff[diff > 0.001].head(10).index
            cmp = pd.DataFrame({"api": api_local.reindex(idx),
                                 "manual": man.reindex(idx),
                                 "diff": diff.reindex(idx)})
            print(f"\n  First mismatches:")
            print(cmp.to_string())

    # ── hourly aggregation comparison ─────────────────────────────────────
    print(f"\n── Hourly aggregation comparison ──")
    hourly_range = pd.date_range(
        start=f"{cfg['start']} 00:00:00",
        end=f"{cfg['end']} 23:00:00",
        freq="h",
    )
    api_hourly = to_hourly_from_raw(api_df, cfg["timezone"], cfg["tip_size"], hourly_range)
    man_hourly = to_hourly_from_raw(
        man_df.rename(columns={"timestamp": "timestamp"}).assign(
            timestamp=pd.to_datetime(man_df["timestamp"]).pipe(
                lambda s: s if s.dt.tz is None
                else s.dt.tz_localize(None)
            )
        ).pipe(lambda df: df.assign(
            timestamp=pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        )),
        cfg["timezone"], cfg["tip_size"], hourly_range,
    )

    h_diff = (api_hourly - man_hourly).abs()
    print(f"  Total mm — API hourly: {api_hourly.sum():.1f}  manual hourly: {man_hourly.sum():.1f}")
    print(f"  Max |hourly diff|: {h_diff.max():.4f} mm")
    print(f"  Hours with diff > 0.01 mm: {(h_diff > 0.01).sum()}")
    print(f"  Hourly Pearson r: {np.corrcoef(api_hourly, man_hourly)[0,1]:.4f}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — pipeline integrity: rebuild hourly from raw and diff vs long CSV
# ════════════════════════════════════════════════════════════════════════════

def cmd_verify(args):
    cfg = REGION_CFG[args.region]

    print(f"\n{'='*70}")
    print(f"SECTION 2 — Pipeline integrity   region={args.region}")
    print(f"{'='*70}")

    # Load built long CSV
    if not cfg["long_csv"].exists():
        sys.exit(f"[!] Long CSV not found: {cfg['long_csv']}")
    print(f"\nLoading built long CSV: {cfg['long_csv']}")
    long = pd.read_csv(cfg["long_csv"])
    long["timestamp"] = pd.to_datetime(long["timestamp"])
    long["station_id"] = long["station_id"].astype(str).str.zfill(6)
    built = long.pivot(index="timestamp", columns="station_id", values="rainfall_mm")
    print(f"  Shape: {built.shape}  (timesteps × stations)")
    print(f"  Time range: {built.index.min()}  →  {built.index.max()}")

    hourly_range = pd.date_range(
        start=f"{cfg['start']} 00:00:00",
        end=f"{cfg['end']} 23:00:00",
        freq="h",
    )

    # Find raw CSVs for stations in the long CSV
    raw_dir = cfg["raw_dir"]
    station_ids = list(built.columns)
    n_check = min(len(station_ids), args.n_stations)
    check_ids = station_ids[:n_check]

    print(f"\nRebuilding hourly for {n_check} stations from raw CSVs ...")

    results = []
    for sid in check_ids:
        raw_path = raw_dir / f"{sid}_rainfall_raw.csv"
        if not raw_path.exists():
            results.append({"station": sid, "status": "RAW_MISSING", "max_diff": np.nan,
                            "total_api": np.nan, "total_built": np.nan, "r": np.nan})
            continue

        api_df = load_raw_api_csv(raw_path)
        rebuilt = to_hourly_from_raw(api_df, cfg["timezone"], cfg["tip_size"], hourly_range)

        # Built values for this station
        if sid not in built.columns:
            results.append({"station": sid, "status": "NOT_IN_LONG", "max_diff": np.nan,
                            "total_api": rebuilt.sum(), "total_built": np.nan, "r": np.nan})
            continue

        built_s = built[sid].reindex(hourly_range).fillna(0)
        diff = (rebuilt - built_s).abs()
        r = float(np.corrcoef(rebuilt, built_s)[0, 1]) if rebuilt.std() > 0 and built_s.std() > 0 else np.nan

        results.append({
            "station": sid,
            "status":  "MATCH" if diff.max() < 0.001 else "MISMATCH",
            "max_diff": float(diff.max()),
            "total_api": float(rebuilt.sum()),
            "total_built": float(built_s.sum()),
            "r": r,
        })

    res = pd.DataFrame(results)
    print(f"\n  Results summary:")
    print(res.to_string(index=False))

    mismatches = res[res["status"] == "MISMATCH"]
    if mismatches.empty:
        print(f"\n✓ All {n_check} stations: rebuilt hourly MATCHES built long CSV (within 0.001 mm).")
    else:
        print(f"\n⚠  {len(mismatches)} station(s) have mismatches — inspect raw CSVs for those stations.")


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — timestamp sanity
# ════════════════════════════════════════════════════════════════════════════

def cmd_qa(args):
    cfg = REGION_CFG[args.region]

    print(f"\n{'='*70}")
    print(f"SECTION 3 — Timestamp sanity   region={args.region}")
    print(f"{'='*70}")

    if not cfg["long_csv"].exists():
        sys.exit(f"[!] Long CSV not found: {cfg['long_csv']}")

    long = pd.read_csv(cfg["long_csv"])
    long["timestamp"] = pd.to_datetime(long["timestamp"])
    long["station_id"] = long["station_id"].astype(str).str.zfill(6)
    built = long.pivot(index="timestamp", columns="station_id", values="rainfall_mm")

    T, N = built.shape
    print(f"\nDataset: {T} timesteps × {N} stations")
    print(f"Time range: {built.index.min()}  →  {built.index.max()}")

    # Hourly regularity
    diffs = built.index.to_series().diff().dropna()
    expected = pd.Timedelta("1h")
    n_gaps = (diffs != expected).sum()
    print(f"\n── Hourly regularity ──")
    print(f"  Expected 1h steps: {len(diffs):,}  |  Non-1h gaps: {n_gaps}")
    if n_gaps:
        bad = built.index[1:][diffs != expected]
        print(f"  Gap timestamps (first 5): {list(bad[:5])}")

    # DST check: if gauge timestamps are timezone-naive local (Australia/Sydney),
    # there should be a DST spring-forward gap and fall-back repeat.
    # In 2022: spring-forward Oct 2 02:00 → 03:00 (missing 02:00 is expected)
    #           fall-back  Apr 3 03:00 → 02:00 (duplicate is handled by tz_localize)
    if args.region in ("sydney", "wagga"):
        apr_gap = pd.Timestamp("2022-04-03 02:00:00")
        oct_gap = pd.Timestamp("2022-10-02 02:00:00")
        print(f"\n── DST check (AEST/AEDT 2022) ──")
        print(f"  Apr-3 02:00 (fall-back — this hour should exist ONCE in local time): "
              f"{'present' if apr_gap in built.index else 'absent'}")
        print(f"  Oct-2 02:00 (spring-forward — this hour should be MISSING in local time): "
              f"{'present' if oct_gap in built.index else 'MISSING (expected)'}")

    # NaN / coverage
    print(f"\n── Coverage / NaN ──")
    nan_frac = built.isna().mean()
    print(f"  Stations with NaN fraction > 5%:  {(nan_frac > 0.05).sum()}")
    print(f"  Stations with NaN fraction > 20%: {(nan_frac > 0.20).sum()}")
    print(f"  Min coverage: {(1-nan_frac.max()):.2%}  "
          f"Median coverage: {(1-nan_frac.median()):.2%}")

    # Value sanity
    print(f"\n── Value sanity ──")
    vals = built.values.flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  Mean mm/h: {vals.mean():.4f}")
    print(f"  Max mm/h:  {vals.max():.2f}")
    print(f"  Wet fraction (>0): {(vals > 0).mean():.4f}")
    print(f"  Percentiles 99/99.9/max: "
          f"{np.percentile(vals, [99, 99.9]).round(2)}  max={vals.max():.2f}")
    # Flag suspiciously large values
    per_station_max = built.max()
    suspicious = per_station_max[per_station_max > 100]
    if len(suspicious):
        print(f"\n  ⚠  Stations with max > 100 mm/h ({len(suspicious)}):")
        print(f"  {dict(suspicious.round(1))}")

    # ── Section 4: Gauge-gauge spatial correlation ─────────────────────────
    print(f"\n{'='*70}")
    print(f"SECTION 4 — Gauge–gauge spatial correlation")
    print(f"{'='*70}")

    if not cfg["meta_csv"].exists():
        print(f"  [!] Metadata not found: {cfg['meta_csv']} — skipping.")
        return

    meta = pd.read_csv(
        cfg["meta_csv"], header=None,
        names=["name", "id", "network", "latitude", "longitude"],
    )
    meta["id"] = meta["id"].astype(str).str.zfill(6)
    meta = meta[meta["id"].isin(built.columns)].reset_index(drop=True)
    print(f"\nStations in metadata & long CSV: {len(meta)}")

    # For each station, find nearest 3 neighbours and compute Pearson r at lags 0, ±1h
    coords = meta[["latitude", "longitude"]].values
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=4, metric="haversine")
    nn.fit(np.radians(coords))
    dists_rad, inds = nn.kneighbors(np.radians(coords))

    # Convert radians to km
    R = 6371.0
    dists_km = dists_rad * R

    lag_rs = {lag: [] for lag in (-2, -1, 0, 1, 2)}
    pair_rows = []
    n_pairs = 0
    for i, row in meta.iterrows():
        sid_a = row["id"]
        if sid_a not in built.columns:
            continue
        a = built[sid_a].dropna()
        for k in range(1, 4):   # skip k=0 (self)
            j = inds[i, k]
            sid_b = meta.iloc[j]["id"]
            dist  = dists_km[i, k]
            if sid_b not in built.columns:
                continue
            b = built[sid_b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 200:
                continue
            av = a.reindex(common)
            bv = b.reindex(common)
            if av.std() == 0 or bv.std() == 0:
                continue
            n_pairs += 1
            for lag in lag_rs:
                if lag == 0:
                    r = float(np.corrcoef(av, bv)[0, 1])
                elif lag > 0:
                    r = float(np.corrcoef(av.iloc[lag:], bv.iloc[:-lag])[0, 1])
                else:
                    r = float(np.corrcoef(av.iloc[:lag], bv.iloc[-lag:])[0, 1])
                lag_rs[lag].append(r)
            if n_pairs <= 5:
                r0 = lag_rs[0][-1]
                pair_rows.append({"A": sid_a, "B": sid_b,
                                  "dist_km": round(dist, 1), "r_lag0": round(r0, 3)})

    print(f"\n  Pairs evaluated: {n_pairs}")
    if pair_rows:
        print(f"\n  Sample pairs (nearest neighbours):")
        print(pd.DataFrame(pair_rows).to_string(index=False))

    print(f"\n  Median Pearson r by lag (CRITICAL: should peak at lag 0):")
    for lag in sorted(lag_rs):
        v = lag_rs[lag]
        if v:
            sign = "←" if lag == max(lag_rs, key=lambda l: np.median(lag_rs[l]) if lag_rs[l] else -1) else ""
            print(f"    lag {lag:+d}h : median r = {np.median(v):.3f}  (n={len(v)})  {sign}")
        else:
            print(f"    lag {lag:+d}h : n/a")

    best_lag = max(lag_rs, key=lambda l: np.median(lag_rs[l]) if lag_rs[l] else -1)
    best_r   = np.median(lag_rs[best_lag]) if lag_rs[best_lag] else float("nan")
    r0       = np.median(lag_rs[0]) if lag_rs[0] else float("nan")

    print(f"\n  Best lag: {best_lag:+d}h  (r={best_r:.3f})")
    if best_lag == 0:
        print(f"  ✓ Correlation peaks at lag 0 — gauge timestamps look correctly aligned.")
    else:
        print(f"  ⚠  Correlation peaks at lag {best_lag:+d}h — gauge timestamps may have a "
              f"timezone shift or processing bug!")
    print(f"  Lag-0 median r = {r0:.3f} (expected > 0.5 for nearby station pairs)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["compare", "verify", "qa", "all"],
                   help="Which check(s) to run")
    p.add_argument("--region", default="sydney", choices=list(REGION_CFG),
                   help="Dataset region (default: sydney)")
    p.add_argument("--station", default=None,
                   help="[compare] Station ID to compare (e.g. 566008)")
    p.add_argument("--manual-csv", default=None,
                   help="[compare] Path to manually-downloaded BoM CSV")
    p.add_argument("--n-stations", type=int, default=10,
                   help="[verify] Number of stations to rebuild-check (default 10)")
    args = p.parse_args()

    if args.mode == "compare":
        if not args.station or not args.manual_csv:
            p.error("compare mode requires --station and --manual-csv")
        cmd_compare(args)
    elif args.mode == "verify":
        cmd_verify(args)
    elif args.mode == "qa":
        cmd_qa(args)
    elif args.mode == "all":
        cmd_verify(args)
        cmd_qa(args)


if __name__ == "__main__":
    main()
