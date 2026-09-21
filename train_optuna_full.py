"""Optuna hyperparameter tuning for the full gauge+radar+satellite GNN.

Strategy
---------
- Single objective : minimise validation station-median RMSE
- Fold 3           : most representative across all three modality experiments
- All params fresh : no values inherited from previous radar-only tuning
- Tuned            : gauge_gauge, radar_gauge, satellite_gauge, is_directed,
                     hidden_channels, num_layers, lr, weight_decay, dropout,
                     weighted_loss_alpha
- 100 trials, TPE sampler, MedianPruner(startup=15, warmup=25)
- Max 50 epochs per trial, early stop on station-median RMSE (patience=7)

Usage
-----
  python train_optuna_full.py --config config_gauge_radar_satellite.yaml
  python train_optuna_full.py --config config_gauge_radar_satellite.yaml --smoketest
"""

import argparse
import copy
import json
import os
import pickle

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.graph.gaugegraphnew import (
    GaugeGraphNew,
    HeterogeneousWeatherGraphDatasetInductive,
)
from src.graph.radargraph_au import AURadarGraph
from src.graph.satellitegraph_au import AUSatelliteGraph
from src.radar.au_preprocessor import AURadarPreprocessor
from src.raingauge.utils import load_raingauge_dataset
from src.sampling.main import stratified_spatial_kfold_dual
from src.satellite.au_preprocessor import AUSatellitePreprocessor
from src.utils import read_config
from training.logic_hetero import bg_predict, train_epoch, validate

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_gauge_radar_satellite.yaml")
parser.add_argument("--n-trials", type=int, default=100)
parser.add_argument("--smoketest", action="store_true",
                    help="3 trials, 5 epochs each — pipeline check only")
args = parser.parse_args()

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLD      = 3   # most representative across gauge-only / gauge+radar / gauge+radar+satellite
DB_PATH   = "optuna_full.db"
STUDY_NAME = f"full_station_median_rmse_fold{FOLD}"
OUT_DIR   = "optuna_results_full"
os.makedirs(OUT_DIR, exist_ok=True)

if args.smoketest:
    N_TRIALS   = 3
    MAX_EPOCHS = 5
    EARLY_STOP = 3
else:
    N_TRIALS   = args.n_trials
    MAX_EPOCHS = 50
    EARLY_STOP = 7

# ── Load base config ──────────────────────────────────────────────────────────
base_config = read_config(args.config)
USE_BG     = base_config["training_params"].get("use_bernoulli_gamma", False)

print(f"Device  : {DEVICE}")
print(f"Config  : {args.config}")
print(f"Fold    : {FOLD}  (most representative)")
print(f"Trials  : {N_TRIALS}  |  Max epochs : {MAX_EPOCHS}  |  Early stop : {EARLY_STOP}")
print(f"DB      : {DB_PATH}")
print(f"Output  : {OUT_DIR}/")

# ── Load raingauge data once ──────────────────────────────────────────────────
dp = base_config["dataset_parameters"]
raingauge_df, meta_df = load_raingauge_dataset(
    rainfall_file=dp["rainfall_file"],
    metadata_file=dp["metadata_file"],
    start=dp["start_year"],
    end=dp["end_year"],
    uptime_threshold=base_config["filters"]["uptime_threshold"],
)
print(f"Raingauge: {raingauge_df.shape[0]} timesteps × {raingauge_df.shape[1]} stations")

# ── Load radar data once ──────────────────────────────────────────────────────
radar_cache = dp.get("radar_cache_path", None)
preprocessor_radar = AURadarPreprocessor(
    dp.get("radar_base_path", "/g/data/rq0/rainfields3"),
    dp.get("radar_id", 71),
    dp.get("radar_stride", 10),
)

if radar_cache and os.path.exists(radar_cache):
    print(f"Loading cached radar from {radar_cache}")
    with open(radar_cache, "rb") as f:
        radar_df_raw = pickle.load(f)
else:
    print("Building radar dataset from raw files…")
    radar_df_raw = preprocessor_radar.build_dataset_range(dp["start_year"], dp["end_year"])
    if radar_cache:
        os.makedirs(os.path.dirname(radar_cache) or ".", exist_ok=True)
        with open(radar_cache, "wb") as f:
            pickle.dump(radar_df_raw, f)
        print(f"Cached radar → {radar_cache}")

radar_graph = AURadarGraph(
    radar_df=radar_df_raw,
    gauge_index=raingauge_df.index,
    node_coords=preprocessor_radar.get_node_coords(),
    grid_shape=preprocessor_radar.grid_shape,
)
print(f"Radar graph: {radar_graph.n_nodes} nodes, grid {radar_graph.grid_shape}")

# ── Load satellite data once ──────────────────────────────────────────────────
sat_cache = dp.get("satellite_cache_path", None)
preprocessor_sat = AUSatellitePreprocessor(
    dp.get("satellite_base_path",
           "/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph")
)

if sat_cache and os.path.exists(sat_cache):
    print(f"Loading cached satellite from {sat_cache}")
    with open(sat_cache, "rb") as f:
        satellite_df_raw = pickle.load(f)
else:
    print("Building satellite dataset from raw files…")
    satellite_df_raw = preprocessor_sat.build_dataset_range(dp["start_year"], dp["end_year"])
    if sat_cache:
        os.makedirs(os.path.dirname(sat_cache) or ".", exist_ok=True)
        with open(sat_cache, "wb") as f:
            pickle.dump(satellite_df_raw, f)
        print(f"Cached satellite → {sat_cache}")

satellite_graph = AUSatelliteGraph(
    satellite_df=satellite_df_raw,
    gauge_index=raingauge_df.index,
    node_coords=preprocessor_sat.get_node_coords(),
    grid_shape=preprocessor_sat.grid_shape,
)
print(f"Satellite graph: {satellite_graph.n_nodes} nodes, grid {satellite_graph.grid_shape}")

# ── K-fold split ──────────────────────────────────────────────────────────────
split_info = stratified_spatial_kfold_dual(
    meta_df,
    seed=base_config["training_params"]["seed"],
    plot=False,
    n_splits=5,
)

# ── Validation helper — station-median RMSE ───────────────────────────────────
_DATA_FEATURE_DIM = 2  # rainfall + validity flag; temporal/LPE cols must not be masked


def validate_station_median_rmse(model, loader, device, use_bg):
    """Station-median RMSE on validation nodes.

    Groups predictions by station index (node position within the graph) and
    returns the median per-station RMSE — consistent with the test_model metric.
    """
    model.eval()
    all_preds      = []
    all_targets    = []
    all_station_ids = []

    with torch.no_grad():
        for batch in loader:
            batch    = batch.to(device)
            x        = batch["raingauge"].x       # [B*N, F]
            y        = batch["raingauge"].y        # [B*N, 1]
            val_mask = batch["raingauge"].mask     # bool [B*N]

            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            x_dict["raingauge"] = x_masked

            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], "edge_attr")
            }

            out  = model(x_dict, batch.edge_index_dict, edge_attr_dict)
            pred = bg_predict(out["raingauge"]) if use_bg else out["raingauge"].clamp(min=0.0)

            pred_v   = pred[val_mask]
            tgt_v    = y[val_mask]
            valid_v  = x[val_mask, 1].bool()   # validity flag — excludes NaN-filled targets
            num_nodes = batch["raingauge"].x.shape[0] // batch.num_graphs

            if valid_v.sum() == 0:
                continue

            station_ids = (val_mask.nonzero(as_tuple=False).squeeze() % num_nodes)

            all_preds.append(pred_v[valid_v].cpu())
            all_targets.append(tgt_v[valid_v].cpu())
            all_station_ids.append(station_ids[valid_v].cpu())

    if not all_preds:
        return float("inf")

    preds      = torch.cat(all_preds).flatten().numpy()
    targets    = torch.cat(all_targets).flatten().numpy()
    station_ids = torch.cat(all_station_ids).flatten().numpy()

    station_rmses = []
    for sid in np.unique(station_ids):
        mask = station_ids == sid
        if mask.sum() < 2:
            continue
        rmse = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
        station_rmses.append(rmse)

    return float(np.median(station_rmses)) if station_rmses else float("inf")


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    config = copy.deepcopy(base_config)

    # ── Search space ─────────────────────────────────────────────────────────
    config["layer_connect"]["gauge_gauge"]     = trial.suggest_int("gauge_gauge", 3, 15)
    config["layer_connect"]["radar_gauge"]     = trial.suggest_int("radar_gauge", 3, 20)
    config["layer_connect"]["satellite_gauge"] = trial.suggest_int("satellite_gauge", 3, 20)
    config["layer_connect"]["is_directed"]     = trial.suggest_categorical("is_directed", [True, False])

    config["model"]["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
    config["model"]["num_layers"]      = trial.suggest_int("num_layers", 2, 8)
    config["model"]["learning_rate"]   = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    config["model"]["weight_decay"]    = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    config["model"]["dropout"]         = trial.suggest_float("dropout", 0.0, 0.3)

    config["training_params"]["weighted_loss_alpha"] = trial.suggest_float("weighted_loss_alpha", 0.0, 1.0)

    print(f"\n[Trial {trial.number}] params: {trial.params}")

    # ── Build fused graph ─────────────────────────────────────────────────────
    try:
        gauge_graph = GaugeGraphNew(
            raingauge_df, meta_df,
            split_info=split_info[FOLD],
            knn=config["layer_connect"]["gauge_gauge"],
            config=config,
        )
        gauge_graph.add_heterodata(
            heterodata_layer=radar_graph.get_radar_heterodata(),
            coords=radar_graph.grid_coords,
            layer_name="radar",
            knn=config["layer_connect"]["radar_gauge"],
        )
        gauge_graph.add_heterodata(
            heterodata_layer=satellite_graph.get_satellite_heterodata(),
            coords=satellite_graph.grid_coords,
            layer_name="satellite",
            knn=config["layer_connect"]["satellite_gauge"],
        )
    except Exception as e:
        print(f"[Trial {trial.number}] Graph build failed: {e}")
        raise optuna.exceptions.TrialPruned()

    train_hetero = gauge_graph.get_train_heterodata()
    val_hetero   = gauge_graph.get_validation_heterodata()

    batch_size = base_config["training_params"]["batch_size"]
    train_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(train_hetero),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(val_hetero),
        batch_size=batch_size,
        shuffle=False,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = GNNInductiveHetero(
        in_channels_dict={"raingauge": 1, "radar": 1, "satellite": 2},
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=3 if USE_BG else 1,
        num_layers=config["model"]["num_layers"],
        edge_types=train_hetero.edge_types,
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=float(config["model"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    wmse_alpha = config["training_params"]["weighted_loss_alpha"]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_median_rmse = float("inf")
    no_improve       = 0

    for epoch in range(MAX_EPOCHS):
        train_epoch(
            model, train_loader, optimizer, DEVICE,
            use_bg=USE_BG,
            weighted_loss_alpha=wmse_alpha,
            use_tweedie=False,
        )

        val_loss = validate(
            model, val_loader, DEVICE,
            use_bg=USE_BG,
            weighted_loss_alpha=wmse_alpha,
            use_tweedie=False,
        )
        scheduler.step(val_loss)

        # Report val loss for pruning (cheap — already computed)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned at epoch {epoch}  val_loss={val_loss:.4f}")
            raise optuna.exceptions.TrialPruned()

        median_rmse = validate_station_median_rmse(model, val_loader, DEVICE, USE_BG)

        if median_rmse < best_median_rmse:
            best_median_rmse = median_rmse
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[Trial {trial.number}] Epoch {epoch:2d}: "
            f"loss={val_loss:.4f}  median_rmse={median_rmse:.4f}  best={best_median_rmse:.4f}"
        )

        if no_improve >= EARLY_STOP:
            print(f"[Trial {trial.number}] Early stop at epoch {epoch}  best={best_median_rmse:.4f}")
            break

    print(f"[Trial {trial.number}] Done — best_median_rmse={best_median_rmse:.4f}")
    return best_median_rmse


# ── Study ─────────────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="minimize",
    study_name=STUDY_NAME,
    storage=f"sqlite:///{DB_PATH}",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=25,
        interval_steps=1,
    ),
)

study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

# ── Save results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEST TRIAL")
print("=" * 60)
best = study.best_trial
print(f"  Station-median RMSE : {best.value:.6f}")
print(f"  Params              : {best.params}")

with open(f"{OUT_DIR}/best_params.json", "w") as f:
    json.dump({"best_station_median_rmse": best.value, "params": best.params}, f, indent=2)
print(f"\nSaved best params → {OUT_DIR}/best_params.json")

trials_df = study.trials_dataframe()
trials_df.to_csv(f"{OUT_DIR}/all_trials.csv", index=False)
print(f"Saved all trials  → {OUT_DIR}/all_trials.csv")

print("\nTop 10 trials by station-median RMSE:")
top10 = (
    trials_df[trials_df["state"] == "COMPLETE"]
    .sort_values("value")
    .head(10)[
        ["number", "value"]
        + [c for c in trials_df.columns if c.startswith("params_")]
    ]
)
print(top10.to_string(index=False))
