import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import read_config

import os

# Number of data features per raingauge node (rainfall value + validity flag).
# LPE columns start at index _DATA_FEATURE_DIM and must NOT be zeroed during masking.
_DATA_FEATURE_DIM = 2

def bernoulli_gamma_loss(pred_raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Bernoulli-Gamma negative log-likelihood for zero-inflated rainfall data.

    Separates the zero-inflation problem into two independent components:
      - Bernoulli : P(rain)       — trained on all samples
      - Gamma     : E[rain|rain>0] — trained only on wet samples

    This avoids the Tweedie tension where 92 % dry-sample gradients compete
    against 8 % wet-sample gradients in a single formula.

    Parameters
    ----------
    pred_raw : Tensor [N, 3]  – raw model outputs (no activation applied)
        col 0 : p_logit   → rain probability  p = sigmoid(p_logit)
        col 1 : mu_raw    → conditional mean  μ = relu(mu_raw).clamp(1e-3)
        col 2 : alpha_raw → Gamma shape       α = relu(alpha_raw).clamp(1e-3)
    target   : Tensor [N] or [N,1] – observed rainfall ≥ 0

    Returns
    -------
    Scalar mean NLL loss.

    Loss per sample
    ---------------
    y = 0 :  -log(1 - p)
    y > 0 :  -log(p) - log_Gamma(y ; α, rate=α/μ)
             where log_Gamma = α·log(α/μ) - lgamma(α) + (α-1)·log(y) - (α/μ)·y
    """
    if target.dim() > 1:
        target = target.squeeze(-1)

    p     = torch.sigmoid(pred_raw[:, 0])
    mu    = F.softplus(pred_raw[:, 1]).clamp(min=1e-3)   # conditional mean — softplus avoids dead gradients
    alpha = F.softplus(pred_raw[:, 2]).clamp(min=1e-3)   # Gamma shape     — softplus avoids dead gradients

    eps      = 1e-7
    dry_mask = (target == 0.0)
    wet_mask = ~dry_mask

    loss_terms = torch.zeros_like(target)

    # ---- Dry branch: -log(1 - p) ----------------------------------------
    if dry_mask.any():
        loss_terms[dry_mask] = -torch.log(1.0 - p[dry_mask] + eps)

    # ---- Wet branch: -log(p) - log_Gamma(y ; α, μ) ----------------------
    if wet_mask.any():
        y_w     = target[wet_mask]
        p_w     = p[wet_mask]
        mu_w    = mu[wet_mask]
        alpha_w = alpha[wet_mask]

        rate = alpha_w / mu_w  # Gamma rate parameter β = α/μ

        log_gamma_pdf = (
            alpha_w * torch.log(rate)
            - torch.lgamma(alpha_w)
            + (alpha_w - 1.0) * torch.log(y_w.clamp(min=eps))
            - rate * y_w
        )

        loss_terms[wet_mask] = -torch.log(p_w + eps) - log_gamma_pdf

    return loss_terms.mean()


def bg_predict(pred_raw: torch.Tensor) -> torch.Tensor:
    """
    Convert raw Bernoulli-Gamma model output → expected rainfall E[y] = p · μ.

    Parameters
    ----------
    pred_raw : Tensor [N, 3]

    Returns
    -------
    Tensor [N, 1]  – predicted rainfall in mm
    """
    p  = torch.sigmoid(pred_raw[:, 0])
    mu = F.softplus(pred_raw[:, 1]).clamp(min=1e-3)
    return (p * mu).unsqueeze(-1)


def tweedie_loss(pred_raw: torch.Tensor, target: torch.Tensor, p: float = 1.6) -> torch.Tensor:
    """
    Tweedie deviance loss for compound Poisson-Gamma data (1 < p < 2).

    Appropriate for Australian hourly rainfall: exact zeros (no-rain) plus a
    continuous positive tail (light-to-heavy rain events).  Reference:
    Hasan & Dunn (2010) — "A Tweedie Compound Poisson Model to Analyse the
    Australian Rainfall Data".

    Parameters
    ----------
    pred_raw : Tensor  – raw model output (unbounded real).  Softplus is applied
                         internally so that the Tweedie mean μ > 0 is guaranteed.
    target   : Tensor  – observed rainfall in mm (non-negative).
    p        : float   – Tweedie power parameter, 1 < p < 2.
                         p = 1.6 recommended by Hasan & Dunn for Australian rainfall.

    Returns
    -------
    Scalar mean Tweedie unit-deviance loss.

    Unit deviance formulas
    ----------------------
    Let p1 = 1 - p  (= -0.6 for p=1.6)
        p2 = 2 - p  (= +0.4 for p=1.6)

    y = 0 :  d = (2 / p2) * μ^p2
    y > 0 :  d = 2 * [ y * (y^p1 - μ^p1) / p1  -  (y^p2 - μ^p2) / p2 ]
    """
    # --- Ensure μ > 0 via softplus, then clamp for numerical safety -----------
    mu = F.softplus(pred_raw).clamp(min=0.01)  # raised from 1e-6: prevents mu^(-0.6) explosion

    p1 = 1.0 - p   # −0.6 for p=1.6
    p2 = 2.0 - p   # +0.4 for p=1.6

    # --- Dry branch (y == 0) --------------------------------------------------
    dry_mask = (target == 0.0)
    d_dry = (2.0 / p2) * mu.pow(p2)

    # --- Wet branch (y > 0) --------------------------------------------------
    # Clamp y away from 0 to avoid y^p1 = y^(−0.6) → ∞ for tiny positives.
    # In practice all positive rainfall values are well above 1e-6 mm.
    y_safe = target.clamp(min=1e-6)
    d_wet = 2.0 * (
        y_safe * (y_safe.pow(p1) - mu.pow(p1)) / p1
        - (y_safe.pow(p2) - mu.pow(p2)) / p2
    )

    d = torch.where(dry_mask, d_dry, d_wet)
    d = d.clamp(max=50.0)  # cap per-sample deviance to prevent extreme values dominating gradients
    return d.mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scheduler=None,
    weighted_loss_alpha: float = 0.0,
    use_tweedie: bool = False,
    tweedie_p: float = 1.6,
    use_bg: bool = False,
):
    """
    Corrected training loop with gradient debugging.
    """
    model.train()
    epoch_losses = []
    charge_bar = tqdm.tqdm(dataloader, desc="training")

    for batch_idx, batch in enumerate(charge_bar):

        optimizer.zero_grad()

        # PyG Batch object - move to device
        batch = batch.to(device)

        # Extract from PyG Batch format
        x = batch['raingauge'].x  # [B*N, F]
        y = batch['raingauge'].y  # [B*N, Tgt]
        # validity: 1 = real gauge reading, 0 = was NaN (filled to 0 in preprocessing).
        # Used to exclude missing-data timesteps from the loss so the model is not
        # trained to predict 0 mm at genuinely unknown timesteps.
        validity = batch['raingauge'].validity  # [B*N]

        edge_index_dict = batch.edge_index_dict
        num_graphs = batch['raingauge'].ptr.size(0) - 1
        num_nodes = x.shape[0] // num_graphs

        edge_attr_dict = {
            edge_type: batch[edge_type].edge_attr
            for edge_type in batch.edge_types
            if hasattr(batch[edge_type], 'edge_attr')
        }

        # Per-node-position backward (mirrors logic_st.py).
        # Calling backward() immediately after each forward pass frees that
        # computation graph before the next forward pass is created, so peak
        # GPU memory is proportional to ONE forward pass rather than to
        # num_nodes forward passes held simultaneously.  Gradients accumulate
        # in p.grad across iterations — equivalent to the single accumulated
        # backward but at a fraction of the memory cost.
        scale = 1.0 / num_nodes
        batch_loss_val = 0.0

        for node_pos in range(num_nodes):
            x_masked = x.clone()
            indices_to_mask = torch.arange(num_graphs, device=device) * num_nodes + node_pos
            x_masked[indices_to_mask, :_DATA_FEATURE_DIM] = 0  # zero data features only; preserve LPE

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict['raingauge'] = x_masked

            out = model(x_dict, edge_index_dict, edge_attr_dict)

            pred_masked = out['raingauge'][indices_to_mask]
            tgt_masked  = y[indices_to_mask]

            # Filter out timesteps where the target gauge had missing data (NaN→0).
            # validity=1 means a real reading was recorded; validity=0 means NaN-filled.
            valid_flag = validity[indices_to_mask] > 0.5
            if valid_flag.sum() == 0:
                continue   # all timesteps in this batch position were missing — skip
            pred_masked = pred_masked[valid_flag]
            tgt_masked  = tgt_masked[valid_flag]

            if use_bg:
                loss = bernoulli_gamma_loss(pred_masked, tgt_masked)
            elif use_tweedie:
                loss = tweedie_loss(pred_masked, tgt_masked, p=tweedie_p)
            elif weighted_loss_alpha > 0.0:
                loss = weighted_mse(pred_masked, tgt_masked, alpha=weighted_loss_alpha)
            else:
                loss = F.mse_loss(pred_masked, tgt_masked)

            (loss * scale).backward()   # frees this graph immediately
            batch_loss_val += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        avg_loss = batch_loss_val / num_nodes
        epoch_losses.append(avg_loss)
        charge_bar.set_postfix({"loss": avg_loss})
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return float(np.mean(epoch_losses))


def validate(
    model,
    dataloader,
    device,
    weighted_loss_alpha: float = 0.0,
    use_tweedie: bool = False,
    tweedie_p: float = 1.6,
    use_bg: bool = False,
):
    """
    Validation loop for PyG batched graph data (inductive setting).

    Key aspects:
    1. Data comes as PyG Batch objects
    2. Features are [B*N, F], already batched and flattened
    3. Mask is [N] - single mask for one graph, replicated across batch
    4. Computes metrics ONLY on validation nodes (where mask=True)
    5. No gradients computed - eval mode
    """
    model.eval()
    epoch_losses = []

    charge_bar = tqdm.tqdm(dataloader, desc="validation")

    with torch.no_grad():
        for batch in charge_bar:
            # PyG Batch object - move to device
            batch = batch.to(device)

            # Extract from PyG Batch format
            x = batch['raingauge'].x  # [B*N, F] - already batched and flattened
            y = batch['raingauge'].y  # [B*N, Tgt] - already batched and flattened
            val_mask = batch['raingauge'].mask  # [N] - single mask for one graph
            validity = batch['raingauge'].validity  # [B*N] - 1=real reading, 0=was NaN
            edge_index_dict = batch.edge_index_dict  # [2, E*B] - offset edge indices
            edge_attr_dict = {
                edge_type: batch[edge_type].edge_attr
                for edge_type in batch.edge_types
                if hasattr(batch[edge_type], 'edge_attr')
            }

            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0  # zero data features only; preserve LPE

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            # Forward pass
            out = model(x_dict, edge_index_dict, edge_attr_dict)  # [B*N, out_channels]

            # Compute loss only on masked nodes that had real readings (validity=1).
            # Excludes NaN-filled zeros from the validation loss signal.
            pred_masked = out['raingauge'][val_mask]
            tgt_masked  = y[val_mask]
            valid_flag  = validity[val_mask] > 0.5
            if valid_flag.sum() > 0:
                pred_masked = pred_masked[valid_flag]
                tgt_masked  = tgt_masked[valid_flag]
            # If all are missing this batch (very unlikely), fall through with full set

            if use_bg:
                loss = bernoulli_gamma_loss(pred_masked, tgt_masked)
                out['raingauge'] = bg_predict(out['raingauge'])
            elif use_tweedie:
                loss = tweedie_loss(pred_masked, tgt_masked, p=tweedie_p)
                out['raingauge'] = F.softplus(out['raingauge']).clamp(min=0.0)
            else:
                out['raingauge'] = out['raingauge'].clamp(min=0.0)
                if weighted_loss_alpha > 0.0:
                    loss = weighted_mse(pred_masked, tgt_masked, alpha=weighted_loss_alpha)
                else:
                    loss = F.mse_loss(pred_masked, tgt_masked)
            epoch_losses.append(loss.item())

    # Compute metrics
    mean_loss = float(np.mean(epoch_losses))

    return mean_loss

def test_model(
    model,
    mapping_df,
    dataloader,
    device,
    fold=0,
    experiment_name="test",
    rain_threshold=0.5,
    use_tweedie: bool = False,
    use_bg: bool = False,
):
    """
    Test loop following the SAME structure as validate():
      - PyG batch format
      - x, y shaped [B*N, F]
      - mask shaped [B*N]
      - station_id shaped [B*N]
      - Computes metrics ONLY on test nodes

    Parameters
    ----------
    rain_threshold : float
        Rainfall threshold (mm) used to binarise predictions and targets
        for computing Precision, Recall, and F1.
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_station_ids = []
    all_validity = []
    epoch_losses = []

    test_bar = tqdm.tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            batch = batch.to(device)
            # ----- Extract inputs from batch -----
            x = batch['raingauge'].x
            y = batch['raingauge'].y
            mask = batch['raingauge'].mask
            validity = batch['raingauge'].validity  # [B*N] 1=real, 0=was NaN
            edge_index = batch.edge_index_dict
            num_graphs = batch['raingauge'].ptr.size(0) - 1
            num_nodes = x.shape[0] // num_graphs

            assert mask.shape[0] == x.shape[0], "Mask and x size mismatch"
            x_masked = x.clone()
            x_masked[mask, :_DATA_FEATURE_DIM] = 0.0  # zero data features only; preserve LPE

            edge_attr_dict = {
                edge_type: batch[edge_type].edge_attr
                for edge_type in batch.edge_types
                if hasattr(batch[edge_type], 'edge_attr')
            }

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            # ----- Model forward -----
            out = model(x_dict, edge_index, edge_attr_dict)
            # Map raw output → rainfall space
            if use_bg:
                out['raingauge'] = bg_predict(out['raingauge'])   # E[y] = p·μ, shape [N,1]
            elif use_tweedie:
                out['raingauge'] = F.softplus(out['raingauge']).clamp(min=0.0)
            else:
                out['raingauge'] = out['raingauge'].clamp(min=0.0)

            # ----- Compute test loss (always MSE for interpretable reporting) -----
            loss = F.mse_loss(out['raingauge'][mask], y[mask])
            epoch_losses.append(loss.item())

            # ----- Collect outputs -----
            all_preds.append(out['raingauge'][mask].detach().cpu())
            all_targets.append(y[mask].detach().cpu())
            all_station_ids.append(
                (mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu()
            )
            all_validity.append(validity[mask].detach().cpu())

            test_bar.set_postfix({"loss": loss.item()})

    # ============================================================
    # === CONCATENATE EVERYTHING
    # ============================================================
    all_preds       = torch.cat(all_preds,       dim=0)
    all_targets     = torch.cat(all_targets,     dim=0)
    all_station_ids = torch.cat(all_station_ids, dim=0)
    all_validity    = torch.cat(all_validity,    dim=0)

    # Remove entries where the target gauge had missing data (NaN→0 filled).
    # These would inflate apparent RMSE and corrupt per-station metrics.
    valid_mask_test = all_validity > 0.5
    n_total   = len(all_preds)
    n_missing = int((~valid_mask_test).sum().item())
    print(f"Filtering {n_missing}/{n_total} test samples where target was NaN "
          f"({100*n_missing/n_total:.1f}%)")
    all_preds       = all_preds[valid_mask_test]
    all_targets     = all_targets[valid_mask_test]
    all_station_ids = all_station_ids[valid_mask_test]

    print("Final aggregated prediction shape:", all_preds.shape)
    print("Final aggregated target shape:", all_targets.shape)
    print("Final aggregated station_id shape:", all_station_ids.shape)

    unique_stations = all_station_ids.unique().tolist()
    print("Total stations in test set:", len(unique_stations))

    # ============================================================
    # === NUMPY ARRAYS (reused everywhere below)
    # ============================================================
    preds_np = all_preds.numpy().flatten()
    targets_np = all_targets.numpy().flatten()
    station_ids_np = all_station_ids.numpy().flatten()

    # ============================================================
    # === GLOBAL REGRESSION METRICS
    # ============================================================
    valid_mask = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, _ = pearsonr(targets_np[valid_mask], preds_np[valid_mask])

    mse = ((all_preds - all_targets) ** 2).mean()
    rmse = torch.sqrt(mse).item()
    mae = compute_mae(preds_np, targets_np)

    print(f"Pearson correlation (Test Nodes): {pearson_r:.4f}")
    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test MAE : {mae:.4f}")

    # ============================================================
    # === GLOBAL CLASSIFICATION METRICS  (NEW)
    # ============================================================
    global_cls = compute_binary_classification_metrics(
        preds_np, targets_np, threshold=rain_threshold
    )
    global_metrics = {"mae": mae, **global_cls}
    print_metrics_summary(global_metrics)

    # ============================================================
    # === TIMESTEP METRICS
    # ============================================================
    # After validity filtering, different stations may have different numbers of
    # valid timesteps, so the flat array is no longer divisible by station count.
    # Fall back to per-station RMSE averaged across stations in that case.
    test_station_count = int(all_station_ids.unique().shape[0])
    try:
        timestep_preds   = all_preds.reshape(-1, test_station_count)
        timestep_targets = all_targets.reshape(-1, test_station_count)
        per_timestep_RMSE = torch.sqrt(
            ((timestep_preds - timestep_targets) ** 2).mean(dim=1)
        )
        timestep_rmse = per_timestep_RMSE.mean().item()
    except RuntimeError:
        # Validity filtering removed some entries unevenly across stations —
        # reshape is no longer valid.  Report NaN; use station_median_rmse instead.
        timestep_rmse = float('nan')
    print(f"Timestep RMSE: {timestep_rmse}")

    # ============================================================
    # === PER-STATION METRICS  (NEW)
    # ============================================================
    per_station = compute_per_station_metrics(
        preds_np, targets_np, station_ids_np, threshold=rain_threshold
    )

    station_rmses = [m["rmse"] for m in per_station.values()]
    station_mean_rmse   = float(np.mean(station_rmses))   if station_rmses else float("nan")
    station_median_rmse = float(np.median(station_rmses)) if station_rmses else float("nan")
    print(f"Station-mean RMSE:   {station_mean_rmse:.4f}")
    print(f"Station-median RMSE: {station_median_rmse:.4f}  (comparable to BRAIN boxplot median)")

    station_rs = [m["pearson_r"] for m in per_station.values() if not np.isnan(m["pearson_r"])]
    station_median_r = float(np.median(station_rs)) if station_rs else float("nan")
    print(f"Station-median r:    {station_median_r:.4f}  (comparable to BRAIN boxplot median)")

    # Save per-station metrics to CSV
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)

    rows = []
    for sid in sorted(per_station.keys()):
        m = per_station[sid]
        rows.append({
            "station_id": sid,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "bias": m["bias"],
            "pearson_r": m["pearson_r"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "support_pos": m["support_pos"],
            "support_neg": m["support_neg"],
        })
    metrics_df = pd.DataFrame(rows)
    csv_path = f"{exp_dir}/per_station_metrics_f{fold}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved per-station metrics CSV → {csv_path}")

    print_metrics_summary(global_metrics, per_station)

    # ============================================================
    # === GLOBAL SCATTER PLOT  (updated annotations)
    # ============================================================
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5)
    max_v = max(np.nanmax(preds_np), np.nanmax(targets_np))
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test Set Performance")
    plt.grid(True)

    text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"Timestep RMSE = {timestep_rmse:.3f}\n"
        f"Station-mean RMSE = {station_mean_rmse:.3f}\n"
        f"Station-median RMSE = {station_median_rmse:.3f}\n"
        f"Station-median r = {station_median_r:.3f}\n"
        f"--- threshold = {rain_threshold} mm ---\n"
        f"Precision = {global_cls['precision']:.3f}\n"
        f"Recall = {global_cls['recall']:.3f}\n"
        f"F1 = {global_cls['f1']:.3f}"
    )
    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )
    plt.savefig(f"{exp_dir}/test_scatter_plot_{fold}.png", dpi=300)
    plt.close()

    # ============================================================
    # === PER-STATION PLOTS  (updated with MAE & F1)
    # ============================================================
    save_dir = f"{exp_dir}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in unique_stations:
        mask_sid = station_ids_np == sid
        preds_sid = preds_np[mask_sid]
        targets_sid = targets_np[mask_sid]

        if len(preds_sid) < 5:
            continue

        station_m = per_station.get(int(sid), None)

        # ----- Scatter -----
        plt.figure(figsize=(7, 7))
        plt.scatter(targets_sid, preds_sid, alpha=0.6)
        max_val = max(preds_sid.max(), targets_sid.max())
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Station {sid} — Actual vs Predicted")
        plt.grid(True)

        if station_m:
            ann = (
                f"MAE = {station_m['mae']:.3f}\n"
                f"F1 = {station_m['f1']:.3f}\n"
                f"Prec = {station_m['precision']:.3f}\n"
                f"Rec = {station_m['recall']:.3f}"
            )
            plt.text(
                0.05, 0.95, ann,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
            )

        plt.savefig(f"{save_dir}/station_{sid}_scatter.png", dpi=250)
        plt.close()

        # ----- Time series -----
        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid, label="Predicted")

        # Draw threshold line for context
        plt.axhline(
            y=rain_threshold, color="gray", linestyle=":", alpha=0.5,
            label=f"Threshold ({rain_threshold} mm)",
        )

        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots in {save_dir}")

    # ============================================================
    # === RETURN RESULTS
    # ============================================================
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "timestep_rmse": timestep_rmse,
        "station_mean_rmse":   station_mean_rmse,
        "station_median_rmse": station_median_rmse,
        "station_median_r":    station_median_r,
        "precision": global_cls["precision"],
        "recall": global_cls["recall"],
        "f1": global_cls["f1"],
        "threshold": rain_threshold,
        "per_station_metrics": per_station,
    }
def weighted_mse(pred, target, alpha: float = 1.0):
    """
    Weighted MSE loss that up-weights high-rainfall timesteps.

    Uses log1p of the target value as the weight signal — stable and
    batch-independent (unlike z-score which collapses on mostly-dry AU batches).

    For target=0   → weight = 1.0
    For target=10  → weight = 1 + alpha*log(11)  ≈ 1 + 2.4*alpha
    For target=50  → weight = 1 + alpha*log(51)  ≈ 1 + 3.9*alpha

    Use alpha=0.0 to recover plain MSE.
    """
    weights = 1.0 + alpha * torch.log1p(target.clamp(min=0.0))
    return (weights * (pred - target) ** 2).mean()

def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error between predictions and targets."""
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    return mean_absolute_error(targets[valid], preds[valid])


def compute_binary_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    pos_label: int = 1,
    zero_division: int = 0,
) -> dict:
    """
    Threshold continuous predictions/targets into binary classes
    (rain >= threshold → 1, else → 0) and compute precision, recall, F1.

    Parameters
    ----------
    preds : np.ndarray      – continuous model predictions
    targets : np.ndarray    – continuous ground-truth values
    threshold : float        – rainfall threshold (mm) for positive class
    pos_label : int          – which class is "positive" (default 1 = rain)
    zero_division : int      – value returned when a metric is undefined

    Returns
    -------
    dict with keys: precision, recall, f1, confusion_matrix, threshold,
                    support_pos, support_neg
    """
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    preds_v = preds[valid]
    targets_v = targets[valid]

    pred_labels = (preds_v >= threshold).astype(int)
    true_labels = (targets_v >= threshold).astype(int)

    precision = precision_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    recall = recall_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    f1 = f1_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
        "support_pos": int(true_labels.sum()),
        "support_neg": int((1 - true_labels).sum()),
    }


# ================================================================
# ===  PER-STATION WRAPPER
# ================================================================

def compute_per_station_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    station_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute MAE and binary classification metrics for each station.

    Parameters
    ----------
    preds : np.ndarray        – flat array of predictions  [N,]
    targets : np.ndarray      – flat array of targets      [N,]
    station_ids : np.ndarray  – flat array of station IDs  [N,]
    threshold : float          – rainfall threshold (mm)

    Returns
    -------
    dict mapping station_id → { mae, precision, recall, f1, ... }
    """
    unique_ids = np.unique(station_ids)
    results = {}

    for sid in unique_ids:
        mask = station_ids == sid
        p = preds[mask]
        t = targets[mask]

        if len(p) < 2:
            continue

        mae = compute_mae(p, t)
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        bias = float(np.mean(p - t))          # positive → over-prediction
        cls = compute_binary_classification_metrics(p, t, threshold=threshold)

        valid = (~np.isnan(p)) & (~np.isnan(t))
        if valid.sum() > 2:
            r, _ = pearsonr(t[valid], p[valid])
            r = float(r)
        else:
            r = float('nan')

        results[int(sid)] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "pearson_r": r,
            **cls,
        }

    return results


def print_metrics_summary(global_metrics: dict, per_station: dict = None):
    """Pretty-print global and (optionally) per-station metrics."""
    print("\n" + "=" * 60)
    print("  GLOBAL METRICS")
    print("=" * 60)
    print(f"  MAE            : {global_metrics['mae']:.4f}")
    print(f"  Threshold      : {global_metrics['threshold']}")
    print(f"  Precision      : {global_metrics['precision']:.4f}")
    print(f"  Recall         : {global_metrics['recall']:.4f}")
    print(f"  F1 Score       : {global_metrics['f1']:.4f}")
    print(f"  Support (pos)  : {global_metrics['support_pos']}")
    print(f"  Support (neg)  : {global_metrics['support_neg']}")
    cm = global_metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    if per_station:
        print("\n" + "=" * 60)
        print("  PER-STATION METRICS")
        print("=" * 60)
        header = f"  {'Station':>8s} | {'MAE':>8s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'Pos':>5s} | {'Neg':>5s}"
        print(header)
        print("  " + "-" * len(header))
        for sid in sorted(per_station.keys()):
            m = per_station[sid]
            print(
                f"  {sid:>8d} | {m['mae']:8.4f} | {m['precision']:6.4f} | "
                f"{m['recall']:6.4f} | {m['f1']:6.4f} | {m['support_pos']:5d} | {m['support_neg']:5d}"
            )
    print("=" * 60)
