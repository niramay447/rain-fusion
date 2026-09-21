"""
Categorical (flood-relevant) verification: POD, FAR, CSI at heavy-rain thresholds.

Addresses reviewer comment: "the evaluation is superficial for the flood motivation —
only aggregate RMSE and Pearson at gauge points are reported, no heavy-rain breakdown,
no categorical score." Aggregate RMSE/Pearson are dominated by the much more common
light-rain regime; POD/FAR/CSI computed at rain-rate thresholds are the standard
categorical-verification protocol for flood/flash-flood utility, both in the
precipitation-nowcasting literature (e.g. Ravuri et al. 2021 "Skilful precipitation
nowcasting"; Sonderby et al. 2020 "MetNet") and in operational meteorology (Roebber
2009 performance diagram). Default thresholds follow the AMS Glossary of Meteorology
rain-rate categories: light <2.5, moderate 2.5-7.6, heavy 7.6-50, violent >50 mm/h.

This codebase's train.py only ever persists aggregate metrics — raw per-node
predictions are computed in training/logic_hetero.py::test_model() and then
discarded. This script re-runs deterministic inference from an already-trained
checkpoint (same data/model reconstruction as mc_dropout_uncertainty.py) and
additionally persists the raw predictions to CSV, so future analyses don't need
another inference pass.

Numbers only — no plots. Outputs per fold + pooled: a CSV table of hits/misses/
false_alarms/POD/FAR/CSI at each threshold, and a JSON summary with the
CSI-maximising threshold.

Usage:
    python flood_verification.py \\
        --config config_gauge_only.yaml \\
        --experiment-dir experiments/20260603_211229_gauge_only \\
        --fold all \\
        --thresholds 1,2.5,5,7.6,10,20,50
"""
from src.sampling.main import stratified_spatial_kfold_dual  # must init first (see train.py)

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.gnn import GNNInductiveHetero
from src.raingauge.utils import load_raingauge_dataset
from src.utils import read_config
from training.logic_hetero import bg_predict
from mc_dropout_uncertainty import build_fold_assets  # reuse fold/dataset/model assembly

# Matches training/logic_hetero.py::test_model() masking: zero rain value (chan 0)
# AND validity flag (chan 1) at the held-out node, not just chan 0 (unlike the
# MC-dropout script, which only masks chan 0 for a different purpose).
_DATA_FEATURE_DIM = 2


def predict_fold(model, dataloader, device, use_bg, use_tweedie):
    """
    Deterministic single-pass inference (dropout off) — same masking and
    forward path as training/logic_hetero.py::test_model(), but returns the
    raw predictions/targets/station_ids instead of only aggregate metrics.
    """
    model.eval()
    preds, targets, station_ids, validity_all = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x = batch['raingauge'].x
            y = batch['raingauge'].y
            mask = batch['raingauge'].mask
            validity = batch['raingauge'].validity
            edge_index = batch.edge_index_dict
            num_graphs = batch['raingauge'].ptr.size(0) - 1
            num_nodes = x.shape[0] // num_graphs

            x_masked = x.clone()
            x_masked[mask, :_DATA_FEATURE_DIM] = 0.0

            edge_attr_dict = {
                edge_type: batch[edge_type].edge_attr
                for edge_type in batch.edge_types
                if hasattr(batch[edge_type], 'edge_attr')
            }
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            x_dict['raingauge'] = x_masked

            out = model(x_dict, edge_index, edge_attr_dict)
            if use_bg:
                out['raingauge'] = bg_predict(out['raingauge'])
            elif use_tweedie:
                out['raingauge'] = F.softplus(out['raingauge']).clamp(min=0.0)
            else:
                out['raingauge'] = out['raingauge'].clamp(min=0.0)

            preds.append(out['raingauge'][mask].detach().cpu())
            targets.append(y[mask].detach().cpu())
            station_ids.append((mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu())
            validity_all.append(validity[mask].detach().cpu())

    preds = torch.cat(preds, dim=0).squeeze(-1)
    targets = torch.cat(targets, dim=0).squeeze(-1)
    station_ids = torch.cat(station_ids, dim=0)
    validity_all = torch.cat(validity_all, dim=0)

    # Drop timesteps where the target gauge had missing data (NaN filled with 0),
    # same filtering test_model() applies before computing metrics.
    valid = validity_all > 0.5
    preds_np = preds[valid].numpy()
    targets_np = targets[valid].numpy()
    station_ids_np = station_ids[valid].numpy()

    finite = np.isfinite(preds_np) & np.isfinite(targets_np)
    return preds_np[finite], targets_np[finite], station_ids_np[finite]


def categorical_scores(preds, targets, thresholds):
    """
    POD, FAR, CSI (+ bias score, success ratio) at each rain-rate threshold.

    POD  (probability of detection) = hits / (hits + misses)          = recall
    FAR  (false alarm ratio)        = false_alarms / (hits + false_alarms) = 1 - precision
    CSI  (critical success index)   = hits / (hits + misses + false_alarms)
    """
    rows = []
    for t in thresholds:
        pred_pos = preds >= t
        true_pos = targets >= t
        tp = int(np.sum(pred_pos & true_pos))
        fp = int(np.sum(pred_pos & ~true_pos))
        fn = int(np.sum(~pred_pos & true_pos))
        tn = int(np.sum(~pred_pos & ~true_pos))

        pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else np.nan
        sr = 1.0 - far if not np.isnan(far) else np.nan
        bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else np.nan

        rows.append({
            "threshold_mm_per_h": t,
            "hits": tp, "misses": fn, "false_alarms": fp, "correct_negatives": tn,
            "pod": pod, "far": far, "csi": csi, "success_ratio": sr,
            "bias_score": bias, "n_events": int(true_pos.sum()),
        })
    return pd.DataFrame(rows)


def best_csi_threshold(preds, targets, t_min=0.1, t_max=50.0, t_step=0.1):
    """Fine sweep over thresholds to find the CSI-maximising operating point."""
    thresholds = np.arange(t_min, t_max + t_step, t_step)
    df = categorical_scores(preds, targets, thresholds)
    df = df.dropna(subset=["csi"])
    if df.empty:
        return None, df
    best_row = df.loc[df["csi"].idxmax()]
    return best_row, df


def _build_datasources(config, start_year, end_year, raingauge_df):
    """Same optional radar/satellite construction as train.py / mc_dropout_uncertainty.py."""
    datasources = config.get('datasources', ['raingauge'])
    use_radar = 'radar' in datasources
    use_satellite = 'satellite' in datasources

    radar_graph = None
    if use_radar:
        from src.graph.radargraph_au import AURadarGraph
        import pickle
        radar_cfg = config['dataset_parameters']
        radar_base = radar_cfg.get('radar_base_path', '/g/data/rq0/rainfields3')
        radar_cache = radar_cfg.get('radar_cache_path', None)
        radar_id = radar_cfg.get('radar_id', 71)
        radar_stride = radar_cfg.get('radar_stride', 10)
        radar_site = radar_cfg.get('radar_site', 'sydney')
        if radar_site == 'wagga':
            from src.radar.wagga_radar_preprocessor import AURadarPreprocessor
        else:
            from src.radar.au_preprocessor import AURadarPreprocessor

        if radar_cache and os.path.exists(radar_cache):
            with open(radar_cache, 'rb') as f:
                radar_df_raw = pickle.load(f)
            preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
        else:
            preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
            radar_df_raw = preprocessor.build_dataset_range(start_year, end_year)

        radar_graph = AURadarGraph(
            radar_df=radar_df_raw, gauge_index=raingauge_df.index,
            node_coords=preprocessor.get_node_coords(), grid_shape=preprocessor.grid_shape,
        )

    satellite_graph = None
    if use_satellite:
        from src.graph.satellitegraph_au import AUSatelliteGraph
        import pickle
        sat_cfg = config['dataset_parameters']
        sat_base = sat_cfg.get(
            'satellite_base_path',
            '/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph',
        )
        sat_cache = sat_cfg.get('satellite_cache_path', None)
        sat_site = sat_cfg.get('satellite_site', 'sydney')
        if sat_site == 'wagga':
            from src.satellite.wagga_satellite_preprocessor import AUSatellitePreprocessor
        else:
            from src.satellite.au_preprocessor import AUSatellitePreprocessor

        if sat_cache and os.path.exists(sat_cache):
            with open(sat_cache, 'rb') as f:
                satellite_df_raw = pickle.load(f)
            preprocessor_sat = AUSatellitePreprocessor(sat_base)
        else:
            preprocessor_sat = AUSatellitePreprocessor(sat_base)
            satellite_df_raw = preprocessor_sat.build_dataset_range(start_year, end_year)

        satellite_graph = AUSatelliteGraph(
            satellite_df=satellite_df_raw, gauge_index=raingauge_df.index,
            node_coords=preprocessor_sat.get_node_coords(), grid_shape=preprocessor_sat.grid_shape,
        )

    return use_radar, use_satellite, radar_graph, satellite_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config.yaml used to train the checkpoint')
    parser.add_argument('--experiment-dir', required=True, help='experiments/<name> dir containing weather_gnn_best_{fold}.pth')
    parser.add_argument('--fold', default='all', help='Fold index, or "all" to run + pool every fold found in --experiment-dir')
    parser.add_argument('--thresholds', default='1,2.5,5,7.6,10,20,50',
                         help='Comma-separated rain-rate thresholds in mm/h for the reported POD/FAR/CSI table. '
                              'Default follows AMS Glossary of Meteorology light/moderate/heavy/violent boundaries.')
    parser.add_argument('--sweep-max', type=float, default=50.0, help='Max threshold (mm/h) for the CSI-optimising sweep')
    parser.add_argument('--sweep-step', type=float, default=0.1, help='Step size (mm/h) for the CSI-optimising sweep')
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read_config(args.config)
    batch_size = config['training_params']['batch_size']
    fold_count = config['training_params']['fold_count']
    use_tweedie = config['training_params'].get('use_tweedie', False)
    use_bg = config['training_params'].get('use_bernoulli_gamma', False)
    log_transform = config['dataset_parameters'].get('log_transform', False)

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        rainfall_file=config['dataset_parameters'].get('rainfall_file', 'all_stations_rainfall_hourly_combined.csv'),
        metadata_file=config['dataset_parameters'].get('metadata_file', 'database/australia/station_metadata.csv'),
        start=start_year, end=end_year, uptime_threshold=uptime_threshold,
    )
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    use_radar, use_satellite, radar_graph, satellite_graph = _build_datasources(
        config, start_year, end_year, raingauge_df
    )

    hidden_channels = config['model']['hidden_channels']
    num_layers = config['model']['num_layers']
    out_channels = 3 if use_bg else 1
    in_channels_dict = {"raingauge": 1}
    if use_radar:
        in_channels_dict["radar"] = 2
    if use_satellite:
        in_channels_dict["satellite"] = 2

    if args.fold == 'all':
        folds = sorted(
            int(fn.split('_')[-1].split('.')[0])
            for fn in os.listdir(args.experiment_dir)
            if fn.startswith('weather_gnn_best_') and fn.endswith('.pth')
        )
    else:
        folds = [int(args.fold)]

    pooled_preds, pooled_targets, pooled_stations, pooled_fold = [], [], [], []

    for fold in folds:
        ckpt_path = os.path.join(args.experiment_dir, f"weather_gnn_best_{fold}.pth")
        print(f"\n=== Fold {fold}: loading {ckpt_path} ===")

        test_loader, edge_types = build_fold_assets(
            config, fold, radar_graph, satellite_graph, raingauge_df,
            raingauge_station_mappings_df, split_info, use_radar, use_satellite,
            log_transform, batch_size,
        )

        model = GNNInductiveHetero(
            in_channels_dict=in_channels_dict, hidden_channels=hidden_channels,
            out_channels=out_channels, num_layers=num_layers, edge_types=edge_types,
            dropout=config['model']['dropout'], hetero_aggr=config['model'].get('hetero_aggr', 'mean'),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        preds, targets, station_ids = predict_fold(model, test_loader, device, use_bg, use_tweedie)
        print(f"n_samples = {len(preds)}")

        pred_df = pd.DataFrame({"station_id": station_ids, "target_mm": targets, "pred_mm": preds})
        pred_csv = os.path.join(args.experiment_dir, f"predictions_f{fold}.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"Saved raw predictions -> {pred_csv}")

        cat_df = categorical_scores(preds, targets, thresholds)
        cat_csv = os.path.join(args.experiment_dir, f"categorical_verification_f{fold}.csv")
        cat_df.to_csv(cat_csv, index=False)
        print(cat_df.to_string(index=False))

        best_row, _sweep_df = best_csi_threshold(
            preds, targets, t_max=args.sweep_max, t_step=args.sweep_step
        )
        if best_row is not None:
            print(f"Best-CSI threshold (fold {fold}): {best_row['threshold_mm_per_h']:.2f} mm/h "
                  f"-> CSI={best_row['csi']:.3f}, POD={best_row['pod']:.3f}, FAR={best_row['far']:.3f}")

        fold_summary = {
            "fold": fold,
            "n_samples": int(len(preds)),
            "categorical_scores": cat_df.to_dict(orient="records"),
            "best_csi_threshold": None if best_row is None else {
                "threshold_mm_per_h": float(best_row["threshold_mm_per_h"]),
                "pod": float(best_row["pod"]), "far": float(best_row["far"]),
                "csi": float(best_row["csi"]), "bias_score": float(best_row["bias_score"]),
            },
        }
        with open(os.path.join(args.experiment_dir, f"categorical_verification_summary_f{fold}.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)

        pooled_preds.append(preds)
        pooled_targets.append(targets)
        pooled_stations.append(station_ids)
        pooled_fold.append(np.full(len(preds), fold))

    if len(folds) > 1:
        preds_all = np.concatenate(pooled_preds)
        targets_all = np.concatenate(pooled_targets)
        stations_all = np.concatenate(pooled_stations)
        folds_all = np.concatenate(pooled_fold)

        pooled_df = pd.DataFrame({
            "fold": folds_all, "station_id": stations_all,
            "target_mm": targets_all, "pred_mm": preds_all,
        })
        pooled_csv = os.path.join(args.experiment_dir, "predictions_all_folds.csv")
        pooled_df.to_csv(pooled_csv, index=False)

        cat_df_all = categorical_scores(preds_all, targets_all, thresholds)
        cat_df_all.to_csv(os.path.join(args.experiment_dir, "categorical_verification_all_folds.csv"), index=False)
        print("\n=== POOLED across all folds ===")
        print(cat_df_all.to_string(index=False))

        best_row_all, _ = best_csi_threshold(
            preds_all, targets_all, t_max=args.sweep_max, t_step=args.sweep_step
        )

        summary_all = {
            "n_folds": len(folds),
            "n_samples": int(len(preds_all)),
            "categorical_scores": cat_df_all.to_dict(orient="records"),
            "best_csi_threshold": None if best_row_all is None else {
                "threshold_mm_per_h": float(best_row_all["threshold_mm_per_h"]),
                "pod": float(best_row_all["pod"]), "far": float(best_row_all["far"]),
                "csi": float(best_row_all["csi"]), "bias_score": float(best_row_all["bias_score"]),
            },
        }
        with open(os.path.join(args.experiment_dir, "categorical_verification_summary_all_folds.json"), "w") as f:
            json.dump(summary_all, f, indent=2)

        print(f"\nSaved pooled verification -> {args.experiment_dir}/categorical_verification_all_folds.csv")
        if best_row_all is not None:
            print(f"Best-CSI threshold (pooled): {best_row_all['threshold_mm_per_h']:.2f} mm/h "
                  f"-> CSI={best_row_all['csi']:.3f}")


if __name__ == '__main__':
    main()
