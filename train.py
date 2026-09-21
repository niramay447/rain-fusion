from src.sampling.main import stratified_spatial_kfold_dual #Dont know why but this has to be initialised first else kernel crashes

import torch
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.transforms import ToUndirected

from src.performance_logger import PerformanceLogger
from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import (
  load_raingauge_dataset
)
from training.logic_hetero import train_epoch, validate, test_model
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive


parser = argparse.ArgumentParser()
parser.add_argument('--smoketest', action='store_true', help='Run 1 fold, 4 epochs only')
parser.add_argument('--config', default='config.yaml', help='Path to config file')
parser.add_argument('--tag', default='', help='Short label to identify this experiment, e.g. exp1_mse_50stn_2022')
parser.add_argument('--max-folds', type=int, default=None,
                    help='Train only the first N folds of the (unchanged) K-fold split. '
                         'Keeps the split identical to the full run so the trained folds '
                         'stay comparable — for fast iteration. Default: all folds.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_config(args.config)
batch_size = config['training_params']['batch_size']
fold_count = 1 if args.smoketest else config['training_params']['fold_count']
# Split is ALWAYS built with n_splits=fold_count (so folds stay comparable);
# n_run_folds only limits how many of those folds we actually build+train.
n_run_folds = fold_count if args.max_folds is None else min(fold_count, args.max_folds)
weighted_loss_alpha = config['training_params']['weighted_loss_alpha']
use_tweedie    = config['training_params'].get('use_tweedie', False)
tweedie_p      = config['training_params'].get('tweedie_p', 1.6)
use_bg         = config['training_params'].get('use_bernoulli_gamma', False)
# log1p the rain/accum feature channel (0) of every source before z-scoring.
# Compresses the heavy rainfall tail so auxiliary radar/sat channels aren't
# outlier-dominated. Features ONLY — target .y stays in physical mm. Default off
# (matches Singapore no-log convention); set True in AU configs to enable.
log_transform = config['dataset_parameters'].get('log_transform', False)

tag = f"_{args.tag}" if args.tag else ""
experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{tag}"
os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

uptime_threshold = config['filters']['uptime_threshold']
start_year = config['dataset_parameters']['start_year']
end_year = config['dataset_parameters']['end_year']
raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
    rainfall_file=config['dataset_parameters'].get('rainfall_file', 'all_stations_rainfall_hourly_combined.csv'),
    metadata_file=config['dataset_parameters'].get('metadata_file', 'database/australia/station_metadata.csv'),
    start=start_year,
    end=end_year,
    uptime_threshold=uptime_threshold,
)

# NaN values are intentionally kept here so that fill_heterodata() can compute
# a correct validity mask (notna()) BEFORE filling with 0.  Filling here would
# make the validity channel permanently all-ones (dead feature).
# fill_heterodata() applies fillna(0) internally for the feature tensor.

split_info = stratified_spatial_kfold_dual(
    raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
)

datasources = config.get('datasources', ['raingauge'])
use_radar = 'radar' in datasources

# ── Radar preprocessing (optional) ──────────────────────────────────────────
radar_graph = None
if use_radar:
    from src.graph.radargraph_au import AURadarGraph
    import pickle, os as _os
    radar_cfg = config['dataset_parameters']
    radar_base = radar_cfg.get('radar_base_path', '/g/data/rq0/rainfields3')
    radar_cache = radar_cfg.get('radar_cache_path', None)
    radar_id    = radar_cfg.get('radar_id', 71)
    radar_stride = radar_cfg.get('radar_stride', 10)
    # Select radar preprocessor by site (default Sydney/71). 'wagga' uses the
    # radar-55 copy with its own projection + full-grid crop; both expose the
    # same AURadarPreprocessor API so nothing else changes.
    radar_site = radar_cfg.get('radar_site', 'sydney')
    if radar_site == 'wagga':
        from src.radar.wagga_radar_preprocessor import AURadarPreprocessor
    else:
        from src.radar.au_preprocessor import AURadarPreprocessor

    if radar_cache and _os.path.exists(radar_cache):
        print(f"Loading cached radar dataset from {radar_cache}")
        with open(radar_cache, 'rb') as f:
            radar_df_raw = pickle.load(f)
        preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
    else:
        print("Building radar dataset from raw Rainfields3 files…")
        preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
        radar_df_raw = preprocessor.build_dataset_range(start_year, end_year)
        if radar_cache:
            _os.makedirs(_os.path.dirname(radar_cache) or '.', exist_ok=True)
            with open(radar_cache, 'wb') as f:
                pickle.dump(radar_df_raw, f)
            print(f"Cached radar dataset → {radar_cache}")

    radar_graph = AURadarGraph(
        radar_df=radar_df_raw,
        gauge_index=raingauge_df.index,
        node_coords=preprocessor.get_node_coords(),
        grid_shape=preprocessor.grid_shape,
    )
    print(f"Radar graph: {radar_graph.n_nodes} nodes, "
          f"grid {radar_graph.grid_shape}")

use_satellite = 'satellite' in datasources

# ── Satellite preprocessing (optional) ──────────────────────────────────────
satellite_graph = None
if use_satellite:
    from src.graph.satellitegraph_au import AUSatelliteGraph
    import pickle, os as _os
    sat_cfg   = config['dataset_parameters']
    sat_base  = sat_cfg.get(
        'satellite_base_path',
        '/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph',
    )
    sat_cache = sat_cfg.get('satellite_cache_path', None)
    # Select satellite preprocessor by site (default Sydney). 'wagga' uses the
    # copy with the Wagga crop + stride; same AUSatellitePreprocessor API.
    sat_site = sat_cfg.get('satellite_site', 'sydney')
    if sat_site == 'wagga':
        from src.satellite.wagga_satellite_preprocessor import AUSatellitePreprocessor
    else:
        from src.satellite.au_preprocessor import AUSatellitePreprocessor

    if sat_cache and _os.path.exists(sat_cache):
        print(f"Loading cached satellite dataset from {sat_cache}")
        with open(sat_cache, 'rb') as f:
            satellite_df_raw = pickle.load(f)
        preprocessor_sat = AUSatellitePreprocessor(sat_base)
    else:
        print("Building satellite dataset from raw CRRPH files…")
        preprocessor_sat = AUSatellitePreprocessor(sat_base)
        satellite_df_raw = preprocessor_sat.build_dataset_range(start_year, end_year)
        if sat_cache:
            _os.makedirs(_os.path.dirname(sat_cache) or '.', exist_ok=True)
            with open(sat_cache, 'wb') as f:
                pickle.dump(satellite_df_raw, f)
            print(f"Cached satellite dataset → {sat_cache}")

    satellite_graph = AUSatelliteGraph(
        satellite_df=satellite_df_raw,
        gauge_index=raingauge_df.index,
        node_coords=preprocessor_sat.get_node_coords(),
        grid_shape=preprocessor_sat.grid_shape,
    )
    print(f"Satellite graph: {satellite_graph.n_nodes} nodes, "
          f"grid {satellite_graph.grid_shape}")

def _log_feature(x):
    """
    log1p the rain/accum data channel (0) only; leave validity (1) + temporal/LPE
    (>=2) untouched. Channel 0 is the rain value for every source (raingauge,
    radar, satellite). No-op unless `log_transform` is enabled. Operates on a
    clone — never mutates the caller's tensor (and never the target .y).
    """
    if not log_transform:
        return x
    x = x.clone()
    x[..., 0] = torch.log1p(x[..., 0].clamp(min=0.0))
    return x


def compute_norm_stats(heterodata):
    """
    Compute per-feature mean and std from a single split's node features.
    Call this on the TRAINING split only to avoid data leakage.

    Shape convention: x is (N, T, F); mean/std are computed over N and T,
    producing one value per feature channel F.

    Matches Singapore train_fused.py compute_norm_stats exactly (plus optional
    log1p on channel 0 when `log_transform` is set — stats are then in log space).
    """
    stats = {}
    for node_type in heterodata.node_types:
        x = _log_feature(heterodata[node_type].x)  # (N, T, F)
        mean = x.mean(dim=(0, 1))                    # (F,)
        std  = x.std(dim=(0, 1)).clamp(min=1e-8)     # (F,)
        stats[node_type] = (mean, std)
        print(f"[Norm] {node_type:12s}  "
              + "  ".join(f"F{f}: μ={mean[f]:.4f} σ={std[f]:.4f}"
                          for f in range(mean.shape[0])))
    return stats


def apply_norm(heterodata, stats):
    """
    Apply precomputed norm stats to a heterodata object.
    Returns a new (cloned) heterodata — never modifies in-place.
    Only touches .x, never .y.

    Matches Singapore train_fused.py apply_norm exactly.
    """
    normed = heterodata.clone()
    for node_type in heterodata.node_types:
        if node_type in stats:
            mean, std = stats[node_type]
            x = _log_feature(heterodata[node_type].x)  # log space iff stats are
            normed[node_type].x = (x - mean) / std
    return normed


gauge_graph_arr = []
for i in range(n_run_folds):
    gauge_graph = GaugeGraphNew(raingauge_df, raingauge_station_mappings_df, split_info=split_info[i], knn=config['layer_connect']['gauge_gauge'], config=config)
    if use_radar:
        gauge_graph.add_heterodata(
            heterodata_layer=radar_graph.get_radar_heterodata(),
            coords=radar_graph.grid_coords,
            layer_name='radar',
            knn=config['layer_connect']['radar_gauge'],
        )
    if use_satellite:
        gauge_graph.add_heterodata(
            heterodata_layer=satellite_graph.get_satellite_heterodata(),
            coords=satellite_graph.grid_coords,
            layer_name='satellite',
            knn=config['layer_connect']['satellite_gauge'],
        )
    gauge_graph_arr.append(gauge_graph)


hidden_channels = config['model']['hidden_channels']
num_layers      = config['model']['num_layers']
out_channels    = 3 if use_bg else 1   # BG needs 3 outputs: p_logit, mu_raw, alpha_raw

in_channels_dict = {"raingauge": 1}
if use_radar:
    in_channels_dict["radar"] = 2  # [accum_mm, valid_flag]
if use_satellite:
    in_channels_dict["satellite"] = 2  # [accum_mm, valid_flag]

model_arr = []
for i in range(n_run_folds):
  model_arr.append(
    GNNInductiveHetero(
      in_channels_dict = in_channels_dict,
      hidden_channels = hidden_channels,
      out_channels=out_channels,
      num_layers = num_layers,
      edge_types = gauge_graph_arr[i].get_train_heterodata().edge_types,
      dropout = config['model']['dropout'],
      hetero_aggr = config['model'].get('hetero_aggr', 'mean'),
    ).to(device=device)
  )

train_loader_arr = []
val_loader_arr = []
test_loader_arr = []
for i in range(n_run_folds):
    train_data = gauge_graph_arr[i].get_train_heterodata()
    val_data   = gauge_graph_arr[i].get_validation_heterodata()
    test_data  = gauge_graph_arr[i].get_test_heterodata()

    # Compute normalisation stats from training split only (no data leakage),
    # then apply same stats to val and test. Matches Singapore train_fused.py.
    norm_stats = compute_norm_stats(train_data)

    train_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(apply_norm(train_data, norm_stats)),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(apply_norm(val_data, norm_stats)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(apply_norm(test_data, norm_stats)),
        batch_size=batch_size,
        shuffle=False,
    )

    train_loader_arr.append(train_loader)
    val_loader_arr.append(val_loader)
    test_loader_arr.append(test_loader)

def train_fold(model, train_loader, val_loader, fold, device="cpu"):
    # CHECK 1: Print initial weights
    print("Training")
    print(f"Device type: {device}")
    first_param = next(model.parameters())
    print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
    )
    # ReduceLROnPlateau: Optuna hyperparameters were tuned WITH this scheduler
    # active (factor=0.5, patience=5). Removing it causes the fixed high LR to
    # oscillate around the minimum, producing periodic validation loss spikes.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    training_loss_arr = []
    validation_loss_arr = []
    early = 0
    mini = float('inf')
    stopping_condition = config['training_params']['early_stop']
    epochs = 0
    total_epochs = 4 if args.smoketest else config['training_params']['epochs']
    print(f"-----FOLD: {fold}-----")
    training_start = time.time()
    for i in range(total_epochs):
        epoch_start = time.time()
        print(f"-----EPOCH: {i + 1}-----")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weighted_loss_alpha=weighted_loss_alpha,
            use_tweedie=use_tweedie,
            tweedie_p=tweedie_p,
            use_bg=use_bg,
        )
        print(train_loss)

        validation_loss = validate(
            model,
            val_loader,
            device,
            weighted_loss_alpha=weighted_loss_alpha,
            use_tweedie=use_tweedie,
            tweedie_p=tweedie_p,
            use_bg=use_bg,
        )
        training_loss_arr.append(train_loss)
        validation_loss_arr.append(validation_loss)
        perf.log_epoch(i, train_loss, validation_loss)
        scheduler.step(validation_loss)
        if mini >= validation_loss:
            mini = validation_loss
            early = 0
            torch.save(
                model.state_dict(), f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth"
            )
        else:
            early += 1
        epochs += 1
        if early >= stopping_condition:
            print("Early stop loss")
            break

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        # CHECK 4: Print gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        print(f"Gradient norm: {total_norm:.6f}")
        epoch_end = time.time()
        print(f"epoch {i} took {epoch_end - epoch_start}")
    training_end = time.time()
    total_time = training_end - training_start
    perf.finalise(total_time)

    print(f"Training took {total_time} seconds over {epochs} epochs")
    plt.plot(training_loss_arr, label="training_loss", color="blue")
    plt.plot(validation_loss_arr, label="validation_loss", color="red")
    plt.legend()
    plt.savefig(f"experiments/{experiment_name}/train_loss_plot_{fold}.png", dpi=300)
    plt.close()

    # Load best checkpoint (saved at minimum validation loss) for testing
    model.load_state_dict(torch.load(f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth"))
    print("✅ Loaded best checkpoint (min val loss) for testing")

import json

all_fold_metrics = []
for i in range(n_run_folds):
    train_fold(model_arr[i], train_loader=train_loader_arr[i], val_loader=val_loader_arr[i], fold=i, device=device)
    fold_metrics = test_model(model_arr[i], raingauge_station_mappings_df, test_loader_arr[i], device, fold=i, experiment_name=experiment_name, use_tweedie=use_tweedie, use_bg=use_bg)

    # Save per-fold metrics to JSON
    metrics_out = {
        "fold":          i,
        "pearson_r":     float(fold_metrics["pearson_r"]),
        "rmse":          float(fold_metrics["rmse"]),
        "mae":           float(fold_metrics["mae"]),
        "timestep_rmse":      float(fold_metrics["timestep_rmse"]),
        "station_mean_rmse":   float(fold_metrics["station_mean_rmse"]),
        "station_median_rmse": float(fold_metrics["station_median_rmse"]),
        "station_median_r":    float(fold_metrics["station_median_r"]),
        "precision":     float(fold_metrics["precision"]),
        "recall":        float(fold_metrics["recall"]),
        "f1":            float(fold_metrics["f1"]),
    }
    all_fold_metrics.append(metrics_out)
    with open(f"experiments/{experiment_name}/test_metrics_f{i}.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

# Save summary across all folds
with open(f"experiments/{experiment_name}/test_metrics_all_folds.json", "w") as f:
    json.dump(all_fold_metrics, f, indent=2)
print(f"\nSaved fold metrics → experiments/{experiment_name}/test_metrics_all_folds.json")