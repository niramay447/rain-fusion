"""
Direct radar/satellite-vs-gauge accuracy baseline (WACV rebuttal).

Reviewer asked whether quality differences in the raw radar/satellite products
(not just the correlation-length argument) explain why fusion is flat on the
Sydney testbed. This script answers that directly: for every held-out test
gauge (same 5-fold spatial-CV split used everywhere else in this repo), it
reads off the RAW radar/satellite value at the pixel nearest that gauge
(1-nearest-neighbour, no gauge information used) and compares it to the gauge
ground truth. No model is trained here — this is a pure sensor-accuracy check,
complementary to the IDW gauge-only baseline (idw_sydney_per_fold.csv) and the
HGNN fusion numbers already in the paper.

Output: radar_direct_sydney_per_fold.csv, satellite_direct_sydney_per_fold.csv
(same column layout as idw_sydney_per_fold.csv for direct comparison).

Coordinate note: AURadarPreprocessor._compute_coords() normally reads one
sample Rainfields3 zip (only available on Gadi, /g/data/rq0) purely to pull
out the fixed 512x512, 0.5 km pixel-centre grid. That grid is a standard,
site-independent Rainfields3 QPE product spec (BoM RAINFIELDS Support Guide
v1.0 sec 4.1: "Single radar grids have a spatial resolution of 0.5 by 0.5 km",
512x512, origin (0,0) = radar site) — reconstructed analytically below as
linspace(-127.75, 127.75, 512) km on both axes. Validated: the resulting
cropped node lat/lon envelope (-34.40..-32.91 N, 149.98..151.45 E) contains
every Sydney gauge (-34.11..-33.61 N, 150.87..151.33 E) with a sane margin,
and node count (952) matches the figure already recorded in project memory
for radar_id=71, stride=10. Satellite coordinates need no such reconstruction
— AUSatellitePreprocessor.get_node_coords() is a pure analytic computation
(no file I/O), so it is used directly.
"""
import pickle

import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

from src.raingauge.utils import load_raingauge_dataset
from src.sampling.main import stratified_spatial_kfold_dual
from src.satellite.au_preprocessor import AUSatellitePreprocessor

# ── knobs (mirrors the IDW baseline notebook cell) ─────────────────────────
FOLD_COUNT = 5
THRESHOLD = 0.5  # mm/h wet/dry threshold for precision/recall/F1
RADAR_CACHE = "database/australia/radar_2022_preprocessed.pkl"
SATELLITE_CACHE = "database/australia/satellite_2022_preprocessed.pkl"
RAINFALL_FILE = "database/australia/rainfall_2022_sydney75_long.csv"
METADATA_FILE = "database/australia/station_metadata_sydney75.csv"

# ── 1. gauge data + identical fold split ────────────────────────────────────
gauge_df, meta_df = load_raingauge_dataset(
    rainfall_file=RAINFALL_FILE, metadata_file=METADATA_FILE,
    start=2022, end=2022, uptime_threshold=0.9,
)
gauge_df.columns = gauge_df.columns.astype(str)
gauge_coords = {str(r.id): (r.latitude, r.longitude) for r in meta_df.itertuples()}

split_info = stratified_spatial_kfold_dual(meta_df, seed=123, plot=False, n_splits=FOLD_COUNT)

gauge_index_utc = gauge_df.index.tz_localize(
    "Australia/Sydney", ambiguous="NaT", nonexistent="shift_forward"
).tz_convert("UTC")

# ── 2. radar: analytic node coords (see module docstring) + cached values ──
_RADAR_PROJ = Proj(
    proj="aea", lat_1=-32.2, lat_2=-35.2, lon_0=151.209, lat_0=-33.7008,
    x_0=0, y_0=0, a=6378137.0, b=6356752.31414,
)
_R0, _R1, _C0, _C1, _RADAR_STRIDE = 100, 434, 30, 309, 10


def radar_node_coords() -> pd.DataFrame:
    x_km = np.linspace(-127.75, 127.75, 512)
    y_km = np.linspace(-127.75, 127.75, 512)
    x_crop = x_km[_C0:_C1 + 1:_RADAR_STRIDE] * 1000.0
    y_crop = y_km[_R0:_R1 + 1:_RADAR_STRIDE] * 1000.0
    xx, yy = np.meshgrid(x_crop, y_crop)
    lon2d, lat2d = _RADAR_PROJ(xx, yy, inverse=True)
    return pd.DataFrame({"latitude": lat2d.flatten(), "longitude": lon2d.flatten()})


radar_coords_df = radar_node_coords()
print(f"[radar] {len(radar_coords_df)} nodes, "
      f"lat [{radar_coords_df.latitude.min():.3f}, {radar_coords_df.latitude.max():.3f}], "
      f"lon [{radar_coords_df.longitude.min():.3f}, {radar_coords_df.longitude.max():.3f}]")
assert radar_coords_df.latitude.min() < meta_df["latitude"].astype(float).min() and \
    radar_coords_df.latitude.max() > meta_df["latitude"].astype(float).max() and \
    radar_coords_df.longitude.min() < meta_df["longitude"].astype(float).min() and \
    radar_coords_df.longitude.max() > meta_df["longitude"].astype(float).max(), \
    "reconstructed radar grid does not enclose all gauges — coordinate reconstruction is wrong"

with open(RADAR_CACHE, "rb") as f:
    radar_raw = pickle.load(f)
radar_lookup = {}
for row in radar_raw.itertuples(index=False):
    ts = row.timestamp
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    radar_lookup[ts] = row.data  # (n_radar_nodes,)

# ── 3. satellite: real (file-free) node coords + cached values ─────────────
sat_preprocessor = AUSatellitePreprocessor()  # no disk I/O in __init__/_compute_coords
sat_coords_df = sat_preprocessor.get_node_coords()
print(f"[satellite] {len(sat_coords_df)} nodes, "
      f"lat [{sat_coords_df.latitude.min():.3f}, {sat_coords_df.latitude.max():.3f}], "
      f"lon [{sat_coords_df.longitude.min():.3f}, {sat_coords_df.longitude.max():.3f}]")

with open(SATELLITE_CACHE, "rb") as f:
    sat_raw = pickle.load(f)
sat_lookup = {}
for row in sat_raw.itertuples(index=False):
    ts = row.timestamp
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    sat_lookup[ts] = row.data  # (n_sat_nodes, 2) = [accum, valid_flag]


# ── 4. nearest-pixel (1-NN, haversine) lookup per gauge, no gauge info used ─
def nearest_node_index(coords_df: pd.DataFrame) -> dict:
    nn = NearestNeighbors(n_neighbors=1, metric="haversine")
    nn.fit(np.radians(coords_df[["latitude", "longitude"]].to_numpy()))
    out = {}
    for sid, (lat, lon) in gauge_coords.items():
        _, idx = nn.kneighbors(np.radians([[lat, lon]]))
        out[sid] = int(idx[0, 0])
    return out


radar_nearest = nearest_node_index(radar_coords_df)
sat_nearest = nearest_node_index(sat_coords_df)

# ── 5. build (T, n_gauges) raw-value + validity matrices, gauge-aligned ────
station_ids = list(gauge_coords.keys())
T = len(gauge_index_utc)

radar_vals = np.full((T, len(station_ids)), np.nan, dtype=np.float64)
sat_vals = np.full((T, len(station_ids)), np.nan, dtype=np.float64)

radar_present = np.array([ts in radar_lookup for ts in gauge_index_utc])
sat_present = np.array([ts in sat_lookup for ts in gauge_index_utc])
print(f"[radar] {radar_present.mean():.1%} of hours have a scan "
      f"({(~radar_present).sum()} entirely missing hours)")

for j, sid in enumerate(station_ids):
    r_idx = radar_nearest[sid]
    s_idx = sat_nearest[sid]
    for t, ts in enumerate(gauge_index_utc):
        r_arr = radar_lookup.get(ts)
        if r_arr is not None:
            radar_vals[t, j] = r_arr[r_idx]
        s_arr = sat_lookup.get(ts)
        if s_arr is not None and s_arr[s_idx, 1] > 0.5:  # valid_flag
            sat_vals[t, j] = s_arr[s_idx, 0]

sat_valid_frac = (~np.isnan(sat_vals)).mean()
print(f"[satellite] {sat_valid_frac:.1%} of gauge-node-hours have a valid retrieval")

radar_pred_df = pd.DataFrame(radar_vals, index=gauge_df.index, columns=station_ids)
sat_pred_df = pd.DataFrame(sat_vals, index=gauge_df.index, columns=station_ids)


# ── 6. per-fold metrics, same layout as idw_sydney_per_fold.csv ────────────
def evaluate(pred_df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for fold in range(FOLD_COUNT):
        test_ids = [str(s) for s in split_info[fold]["ml"]["test"]]
        fp, ft, st_rmse, st_r = [], [], [], []
        for tid in test_ids:
            pred = pred_df[tid].to_numpy(dtype=float)
            targ = gauge_df[tid].to_numpy(dtype=float)
            m = (~np.isnan(pred)) & (~np.isnan(targ))
            if m.sum() < 2:
                continue
            p, t = pred[m], targ[m]
            fp.append(p); ft.append(t)
            st_rmse.append(np.sqrt(np.mean((p - t) ** 2)))
            st_r.append(pearsonr(p, t)[0] if p.std() > 0 and t.std() > 0 else np.nan)
        p = np.concatenate(fp); t = np.concatenate(ft)
        prec, rec, f1, _ = precision_recall_fscore_support(
            t >= THRESHOLD, p >= THRESHOLD, average="binary", zero_division=0)
        rows.append({
            "fold": fold,
            "pearson_r": pearsonr(p, t)[0],
            "rmse": np.sqrt(np.mean((p - t) ** 2)),
            "mae": np.mean(np.abs(p - t)),
            "station_mean_rmse": float(np.mean(st_rmse)),
            "station_median_rmse": float(np.median(st_rmse)),
            "station_median_r": float(np.nanmedian(st_r)),
            "precision": prec, "recall": rec, "f1": f1,
            "n_test_stations": len(st_rmse),
            "n_predictions": len(p),
        })
    per_fold = pd.DataFrame(rows)
    summary = per_fold.drop(columns=["fold"]).mean().to_frame("mean_over_folds").T
    print(f"\n── {label}: per fold ──"); print(per_fold.round(4).to_string(index=False))
    print(f"── {label}: mean over folds ──"); print(summary.round(4).to_string(index=False))
    return per_fold


radar_per_fold = evaluate(radar_pred_df, "RADAR (nearest pixel) vs gauge")
radar_per_fold.to_csv("radar_direct_sydney_per_fold.csv", index=False)

sat_per_fold = evaluate(sat_pred_df, "SATELLITE (nearest pixel) vs gauge")
sat_per_fold.to_csv("satellite_direct_sydney_per_fold.csv", index=False)
