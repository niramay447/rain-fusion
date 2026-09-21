"""
Kriging with External Drift (KED, radar as drift variable) baseline for the
Sydney testbed — same 5-fold spatial-CV split and metric layout as
idw_sydney_per_fold.csv and radar_direct_sydney_per_fold.csv, for direct
comparison.

This replaces benchmarks/models/kriging.py's kriging_external_drift() usage
pattern for Sydney: that function (and the "Kriging interpolation with rain
gauge" cell in benchmarks.ipynb that calls it) hardcodes Singapore's grid
origin (103.6E, 1.51N) when converting a station's lat/lon into a row/col
index into the kriged grid. Sydney gauges sit at roughly (151E, -34N), so
that lookup would index thousands of cells out of bounds. Here we sidestep
grid-index math entirely by calling pykrige's `execute('points', ...)`,
which solves the kriging system directly at the test gauges' coordinates.

Radar node coordinates are reconstructed analytically (no raw Gadi files
needed locally) — see run_radar_satellite_direct.py's module docstring for
the derivation and validation of this approach; reused verbatim here.
"""
import pickle
import time

import numpy as np
import pandas as pd
from pyproj import Proj
from pykrige.uk import UniversalKriging
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support

from src.raingauge.utils import load_raingauge_dataset
from src.sampling.main import stratified_spatial_kfold_dual

# ── knobs (mirrors the IDW / direct-sensor baseline scripts) ───────────────
FOLD_COUNT = 5
THRESHOLD = 0.5          # mm/h wet/dry threshold for precision/recall/F1
VARIOGRAM_MODEL = "linear"   # matches kriging_external_drift()'s own default
MIN_TRAIN_GAUGES = 5     # skip timesteps with too few valid training gauges
RADAR_CACHE = "database/australia/radar_2022_preprocessed.pkl"
RAINFALL_FILE = "database/australia/rainfall_2022_sydney75_long.csv"
METADATA_FILE = "database/australia/station_metadata_sydney75.csv"
OUT_CSV = "ked_sydney_per_fold.csv"

LIMIT_TIMESTEPS = None  # set to an int for a quick timing/smoke run
START_OFFSET = 0        # set >0 to smoke-test a window later in the year (wet season)

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

# ── 2. radar: analytic node coords + cached values (see module docstring) ──
_RADAR_PROJ = Proj(
    proj="aea", lat_1=-32.2, lat_2=-35.2, lon_0=151.209, lat_0=-33.7008,
    x_0=0, y_0=0, a=6378137.0, b=6356752.31414,
)
_R0, _R1, _C0, _C1, _RADAR_STRIDE = 100, 434, 30, 309, 10
_GRID_SHAPE = (len(range(_R0, _R1 + 1, _RADAR_STRIDE)), len(range(_C0, _C1 + 1, _RADAR_STRIDE)))  # (34, 28)


def radar_grid_axes():
    """Analytic lon/lat 2D grids for the cropped radar node grid, shape (H, W) each."""
    x_km = np.linspace(-127.75, 127.75, 512)
    y_km = np.linspace(-127.75, 127.75, 512)
    x_crop = x_km[_C0:_C1 + 1:_RADAR_STRIDE] * 1000.0
    y_crop = y_km[_R0:_R1 + 1:_RADAR_STRIDE] * 1000.0
    xx, yy = np.meshgrid(x_crop, y_crop)  # (H, W)
    lon2d, lat2d = _RADAR_PROJ(xx, yy, inverse=True)
    return lon2d, lat2d  # (H, W) each


lon2d, lat2d = radar_grid_axes()
H, W = _GRID_SHAPE
assert lon2d.shape == (H, W)

# pykrige's external-drift grid requires 1D monotonic axes; the Albers
# projection is not perfectly axis-aligned in lat/lon, so approximate with
# the mean lon per column / mean lat per row (same "assume near-regular
# grid" approximation the Singapore implementation makes via rasterio
# bounds/transform).
drift_x = lon2d.mean(axis=0)   # (W,) increasing west->east
drift_y = lat2d.mean(axis=1)   # (H,) — need increasing order for pykrige
row_order = np.argsort(drift_y)
drift_y_sorted = drift_y[row_order]

print(f"[radar] grid {H}x{W}, lon [{drift_x.min():.3f}, {drift_x.max():.3f}], "
      f"lat [{drift_y.min():.3f}, {drift_y.max():.3f}]")
assert drift_y.min() < meta_df["latitude"].astype(float).min() and \
    drift_y.max() > meta_df["latitude"].astype(float).max() and \
    drift_x.min() < meta_df["longitude"].astype(float).min() and \
    drift_x.max() > meta_df["longitude"].astype(float).max(), \
    "reconstructed radar grid does not enclose all gauges — coordinate reconstruction is wrong"

with open(RADAR_CACHE, "rb") as f:
    radar_raw = pickle.load(f)
radar_lookup = {}
for row in radar_raw.itertuples(index=False):
    ts = row.timestamp
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    radar_lookup[ts] = row.data.reshape(H, W)  # row-major, matches lon2d/lat2d

# ── 3. per-timestep KED fit, evaluated directly at held-out gauge points ───
station_ids = list(gauge_coords.keys())
N_TOTAL = len(gauge_index_utc)
T = (N_TOTAL - START_OFFSET) if LIMIT_TIMESTEPS is None else min(LIMIT_TIMESTEPS, N_TOTAL - START_OFFSET)

results_per_fold = {}
t0 = time.time()
for fold in range(FOLD_COUNT):
    train_ids = [str(s) for s in split_info[fold]["statistical"]["train"]]
    test_ids = [str(s) for s in split_info[fold]["ml"]["test"]]
    test_lons = np.array([gauge_coords[s][1] for s in test_ids])
    test_lats = np.array([gauge_coords[s][0] for s in test_ids])

    pred = np.full((T, len(test_ids)), np.nan, dtype=np.float64)
    n_fitted, n_skipped_radar, n_skipped_gauges, n_failed = 0, 0, 0, 0

    for t in range(T):
        t_real = t + START_OFFSET
        ts = gauge_index_utc[t_real]
        radar_frame = radar_lookup.get(ts)
        if radar_frame is None:
            n_skipped_radar += 1
            continue

        row = gauge_df.iloc[t_real]
        vals = row[train_ids].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() < MIN_TRAIN_GAUGES or np.count_nonzero(vals[valid]) < 1:
            n_skipped_gauges += 1
            continue

        train_lons = np.array([gauge_coords[s][1] for s, v in zip(train_ids, valid) if v])
        train_lats = np.array([gauge_coords[s][0] for s, v in zip(train_ids, valid) if v])
        train_vals = vals[valid]

        try:
            model = UniversalKriging(
                x=train_lons, y=train_lats, z=train_vals,
                variogram_model=VARIOGRAM_MODEL,
                drift_terms=["external_Z"],
                external_drift=radar_frame[row_order, :],
                external_drift_x=drift_x,
                external_drift_y=drift_y_sorted,
                pseudo_inv=True,
            )
            z, _ = model.execute("points", test_lons, test_lats)
            pred[t] = np.maximum(0.0, np.asarray(z))
            n_fitted += 1
        except Exception:
            n_failed += 1
            continue

    print(f"[fold {fold}] fitted {n_fitted}/{T} timesteps "
          f"(skipped: {n_skipped_radar} no-radar, {n_skipped_gauges} too-few-gauges, {n_failed} fit-errors) "
          f"— {time.time() - t0:.1f}s elapsed total")

    pred_df = pd.DataFrame(pred, index=gauge_df.index[START_OFFSET:START_OFFSET + T], columns=test_ids)
    results_per_fold[fold] = pred_df

# ── 4. per-fold metrics, same layout as idw_sydney_per_fold.csv ────────────
rows = []
for fold, pred_df in results_per_fold.items():
    test_ids = pred_df.columns.tolist()
    fp, ft, st_rmse, st_r = [], [], [], []
    for tid in test_ids:
        pred_arr = pred_df[tid].to_numpy(dtype=float)
        targ = gauge_df[tid].reindex(pred_df.index).to_numpy(dtype=float)
        m = (~np.isnan(pred_arr)) & (~np.isnan(targ))
        if m.sum() < 2:
            continue
        p, t_ = pred_arr[m], targ[m]
        fp.append(p); ft.append(t_)
        st_rmse.append(np.sqrt(np.mean((p - t_) ** 2)))
        st_r.append(pearsonr(p, t_)[0] if p.std() > 0 and t_.std() > 0 else np.nan)
    if not fp:
        continue
    p = np.concatenate(fp); t_ = np.concatenate(ft)
    prec, rec, f1, _ = precision_recall_fscore_support(
        t_ >= THRESHOLD, p >= THRESHOLD, average="binary", zero_division=0)
    rows.append({
        "fold": fold,
        "pearson_r": pearsonr(p, t_)[0],
        "rmse": np.sqrt(np.mean((p - t_) ** 2)),
        "mae": np.mean(np.abs(p - t_)),
        "station_mean_rmse": float(np.mean(st_rmse)),
        "station_median_rmse": float(np.median(st_rmse)),
        "station_median_r": float(np.nanmedian(st_r)),
        "precision": prec, "recall": rec, "f1": f1,
        "n_test_stations": len(st_rmse),
        "n_predictions": len(p),
    })

ked_per_fold = pd.DataFrame(rows)
ked_summary = ked_per_fold.drop(columns=["fold"]).mean().to_frame("mean_over_folds").T
ked_per_fold.to_csv(OUT_CSV, index=False)

print("\n── KED (radar drift): per fold ──"); print(ked_per_fold.round(4).to_string(index=False))
print("── KED (radar drift): mean over folds ──"); print(ked_summary.round(4).to_string(index=False))
print(f"\nSaved → {OUT_CSV}")
print(f"Total wall time: {time.time() - t0:.1f}s")
