"""
Optuna hyperparameter tuning for the Australian rainfall GNN.

Strategy
--------
- Single objective: minimise validation RMSE (point-prediction quality)
- BG-NLL is still used as the training loss; RMSE is only the Optuna objective
- Fold 1 only (most representative fold — closest to cross-fold mean RMSE)
- Max 30 epochs, early_stop=5 — fast enough for 50 trials
- TPE sampler (efficient for ~50 trials)
- MedianPruner on per-epoch BG-NLL — kills clearly bad trials early
- Base config: config_exp7_temporal.yaml (BG + LPE + hour + month encoding)

Usage
-----
  python3 train_optuna.py                     # 50 trials
  python3 train_optuna.py --n-trials 20       # quick test
  python3 train_optuna.py --smoketest         # 3 trials, 4 epochs each
"""

import argparse
import copy
import json
import os

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
from src.raingauge.utils import load_raingauge_dataset
from src.sampling.main import stratified_spatial_kfold_dual
from src.utils import read_config
from training.logic_hetero import bernoulli_gamma_loss, bg_predict, train_epoch, validate

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",    default="config_exp7_temporal.yaml")
parser.add_argument("--n-trials",  type=int, default=50)
parser.add_argument("--smoketest", action="store_true",
                    help="3 trials, 4 epochs each — pipeline check only")
args = parser.parse_args()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLD        = 1          # fold 1 is closest to cross-fold mean RMSE — most representative
DB_PATH     = "optuna_raingnn_v2.db"
STUDY_NAME  = "raingnn_rmse_fold1"
OUT_DIR     = "optuna_results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

if args.smoketest:
    N_TRIALS   = 3
    MAX_EPOCHS = 4
    EARLY_STOP = 2
else:
    N_TRIALS   = args.n_trials
    MAX_EPOCHS = 30
    EARLY_STOP = 5

print(f"Device : {DEVICE}")
print(f"Fold   : {FOLD}  (most representative — closest to cross-fold mean RMSE)")
print(f"Trials : {N_TRIALS}  |  Max epochs : {MAX_EPOCHS}  |  Early stop : {EARLY_STOP}")
print(f"Objective: validation RMSE  (training loss remains BG-NLL)")

# ── Load data once (shared across all trials) ─────────────────────────────────
base_config = read_config(args.config)

raingauge_df, meta_df = load_raingauge_dataset(
    rainfall_file=base_config["dataset_parameters"].get(
        "rainfall_file", "all_stations_rainfall_hourly_combined.csv"
    ),
    metadata_file=base_config["dataset_parameters"].get(
        "metadata_file", "database/australia/station_metadata.csv"
    ),
    start=base_config["dataset_parameters"]["start_year"],
    end=base_config["dataset_parameters"]["end_year"],
    uptime_threshold=base_config["filters"]["uptime_threshold"],
)

split_info = stratified_spatial_kfold_dual(
    meta_df, seed=base_config["training_params"]["seed"], plot=False, n_splits=5
)

print(f"Dataset loaded: {raingauge_df.shape[0]} timesteps × {raingauge_df.shape[1]} stations")


# ── RMSE on validation set ────────────────────────────────────────────────────
_DATA_FEATURE_DIM = 2  # rainfall value + validity flag (cols 0-1); matches logic_hetero.py

def validate_rmse(model, loader, device):
    """Compute RMSE on a validation loader. Uses same masking as validate()."""
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            x        = batch['raingauge'].x       # [B*N, F]
            y        = batch['raingauge'].y        # [B*N, 1]
            val_mask = batch['raingauge'].mask     # boolean [B*N]

            # Zero data features for masked nodes (same as validate)
            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], 'edge_attr')
            }

            out  = model(x_dict, batch.edge_index_dict, edge_attr_dict)
            pred = bg_predict(out['raingauge'])    # E[y] = p·μ, shape [N, 1]

            pred_masked = pred[val_mask]
            tgt_masked  = y[val_mask]
            y_valid     = x[val_mask, 1].bool()   # validity flag from original x

            if y_valid.sum() == 0:
                continue

            all_preds.append(pred_masked[y_valid].cpu())
            all_targets.append(tgt_masked[y_valid].cpu())

    if not all_preds:
        return float("inf")

    preds   = torch.cat(all_preds).flatten()
    targets = torch.cat(all_targets).flatten()
    return torch.sqrt(F.mse_loss(preds, targets)).item()


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    config = copy.deepcopy(base_config)

    # ── Hyperparameter search space ───────────────────────────────────────────
    config["model"]["hidden_channels"] = trial.suggest_categorical(
        "hidden_channels", [16, 32, 64, 128]
    )
    config["model"]["num_layers"] = trial.suggest_int("num_layers", 2, 8)
    config["model"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    config["model"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-8, 1e-4, log=True
    )
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    config["training_params"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [128, 256, 512]
    )
    config["layer_connect"]["gauge_gauge"] = trial.suggest_int("gauge_gauge", 3, 10)

    print(f"\n[Trial {trial.number}] params: {trial.params}")

    # ── Build graph for this trial's knn setting ──────────────────────────────
    try:
        gauge_graph = GaugeGraphNew(
            raingauge_df, meta_df,
            split_info=split_info[FOLD],
            knn=config["layer_connect"]["gauge_gauge"],
            config=config,
        )
    except Exception as e:
        print(f"[Trial {trial.number}] Graph build failed: {e}")
        raise optuna.exceptions.TrialPruned()

    train_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(gauge_graph.get_train_heterodata()),
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
    )
    val_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(gauge_graph.get_validation_heterodata()),
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    use_bg = config["training_params"].get("use_bernoulli_gamma", False)
    out_channels = 3 if use_bg else 1

    model = GNNInductiveHetero(
        in_channels_dict={"raingauge": 1},
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=out_channels,
        num_layers=config["model"]["num_layers"],
        edge_types=gauge_graph.get_train_heterodata().edge_types,
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["weight_decay"],
    )
    # Scheduler steps on BG-NLL (training signal); objective uses RMSE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_rmse  = float("inf")
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        train_epoch(
            model, train_loader, optimizer, DEVICE,
            use_bg=use_bg,
            weighted_loss_alpha=config["training_params"].get("weighted_loss_alpha", 0.0),
            use_tweedie=config["training_params"].get("use_tweedie", False),
        )

        # BG-NLL for scheduler + pruning signal
        val_nll = validate(
            model, val_loader, DEVICE,
            use_bg=use_bg,
            weighted_loss_alpha=config["training_params"].get("weighted_loss_alpha", 0.0),
            use_tweedie=config["training_params"].get("use_tweedie", False),
        )
        scheduler.step(val_nll)

        # Report BG-NLL to pruner (correlated with RMSE, cheap to compute)
        trial.report(val_nll, epoch)
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned at epoch {epoch}, val_nll={val_nll:.4f}")
            raise optuna.exceptions.TrialPruned()

        # Compute RMSE for early stopping and final objective
        val_rmse = validate_rmse(model, val_loader, DEVICE)

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            no_improve = 0
        else:
            no_improve += 1

        print(f"[Trial {trial.number}] Epoch {epoch}: NLL={val_nll:.4f}, RMSE={val_rmse:.4f}, best_rmse={best_rmse:.4f}")

        if no_improve >= EARLY_STOP:
            print(f"[Trial {trial.number}] Early stop at epoch {epoch}, best_rmse={best_rmse:.4f}")
            break

    print(f"[Trial {trial.number}] Finished. best_rmse={best_rmse:.4f}")
    return best_rmse


# ── Study ─────────────────────────────────────────────────────────────────────
storage = f"sqlite:///{DB_PATH}"

study = optuna.create_study(
    direction="minimize",
    study_name=STUDY_NAME,
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # don't prune until 5 trials completed
        n_warmup_steps=3,      # don't prune before epoch 3
        interval_steps=1,
    ),
)

study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

# ── Save results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEST TRIAL")
print("=" * 60)
best = study.best_trial
print(f"  Value (val RMSE) : {best.value:.6f}")
print(f"  Params           : {best.params}")

# Best params JSON
best_out = {"best_val_rmse": best.value, "params": best.params}
with open(f"{OUT_DIR}/best_params.json", "w") as f:
    json.dump(best_out, f, indent=2)
print(f"\nSaved best params → {OUT_DIR}/best_params.json")

# All trials CSV
trials_df = study.trials_dataframe()
trials_df.to_csv(f"{OUT_DIR}/all_trials.csv", index=False)
print(f"Saved all trials  → {OUT_DIR}/all_trials.csv")

# Print top 10 trials
print("\nTop 10 trials by val RMSE:")
top10 = (
    trials_df[trials_df["state"] == "COMPLETE"]
    .sort_values("value")
    .head(10)[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]]
)
print(top10.to_string(index=False))
