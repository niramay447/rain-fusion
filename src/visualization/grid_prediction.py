"""
grid_prediction.py
==================
Produce a regular 1 km × 1 km rainfall grid by augmenting the trained
heterogeneous GNN graph with virtual grid-point nodes.

Core idea
---------
Grid points are inserted as new raingauge nodes with zeroed data features
(rainfall value + validity flag).  The model predicts their rainfall from
the surrounding real gauge nodes and, transitively, from radar / satellite
via those gauges.  LPE is recomputed over the full augmented graph so grid
nodes get consistent positional encodings.

Exported symbols
----------------
generate_grid_coords        – build a lon/lat grid DataFrame within bounds
predict_on_grid             – run the model over every timestep; return [T, rows, cols]
plot_rainfall_grid          – visualise a single timestep as a spatial heatmap
plot_rainfall_sequence      – plot a row of N timestep snapshots in one figure
animate_rainfall_grid       – animate the grid as a time series
predict_on_grid_st          – grid inference for the ST model variant
plot_source_comparison      – multi-row × multi-column grid comparing sources / models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import torch

from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.visualization.main import visualise_with_basemap

# Number of data features per raingauge node – must match logic_hetero._DATA_FEATURE_DIM
_DATA_FEATURE_DIM = 2  # [rainfall_value, validity_flag]; LPE columns start at index 2


# ---------------------------------------------------------------------------
# 1.  Grid generation
# ---------------------------------------------------------------------------

def generate_grid_coords(
    bounds: dict,
    resolution_km: float = 1.0,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    """
    Return a regular grid of (longitude, latitude) points within *bounds*
    at the requested *resolution_km* spacing.

    Parameters
    ----------
    bounds       : dict  {'left', 'right', 'top', 'bottom'}  in decimal degrees
    resolution_km: float  grid spacing in kilometres

    Returns
    -------
    grid_coords  : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape   : (n_rows, n_cols)  – rows = latitude axis, cols = longitude axis
    """
    lat_centre = (bounds['top'] + bounds['bottom']) / 2.0
    lat_step = resolution_km / 111.0
    lon_step = resolution_km / (111.32 * np.cos(np.radians(lat_centre)))

    lats = np.arange(bounds['bottom'], bounds['top'],  lat_step)
    lons = np.arange(bounds['left'],   bounds['right'], lon_step)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_coords = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude':  lat_grid.flatten(),
    })
    return grid_coords, (len(lats), len(lons))


# ---------------------------------------------------------------------------
# 2.  Graph augmentation + inference
# ---------------------------------------------------------------------------

def predict_on_grid(
    model,
    heterodata: HeteroData,
    mapping_df: pd.DataFrame,
    bounds: dict,
    resolution_km: float = 1.0,
    knn_gauge: int = 5,
    device: str = 'cpu',
    include_lpe: bool = True,
    lpe_k: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int]]:
    """
    Predict rainfall on a regular grid for every timestep in *heterodata*.

    The function augments the test graph by appending virtual grid nodes as
    new raingauge nodes with zeroed data features.  Each grid node is
    connected to its *knn_gauge* nearest real gauges.  LPE is optionally
    recomputed for the full augmented graph.

    Parameters
    ----------
    model       : trained GNNInductiveHetero (eval mode will be set internally)
    heterodata  : normalised HeteroData returned by GaugeGraphNew.get_test_heterodata()
                  (shape: [N, T, F] tensors)
    mapping_df  : DataFrame with columns ['longitude', 'latitude', 'id'] for real gauges
    bounds      : dict  {'left', 'right', 'top', 'bottom'}
    resolution_km : float  grid spacing in km
    knn_gauge   : int  nearest real gauges to connect to each grid node
    device      : str  torch device string
    include_lpe : bool  recompute Laplacian PE for the augmented graph
    lpe_k       : int  number of LPE eigenvectors (must match model training)

    Returns
    -------
    predictions : np.ndarray  shape [T, n_rows, n_cols]  (in original rainfall units)
    grid_coords : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape  : (n_rows, n_cols)
    """
    model.eval()

    grid_coords, grid_shape = generate_grid_coords(bounds, resolution_km)
    n_grid = len(grid_coords)
    n_real = heterodata['raingauge'].x.shape[0]
    T      = heterodata['raingauge'].x.shape[1]
    F      = heterodata['raingauge'].x.shape[2]

    # ------------------------------------------------------------------
    # Step 1 – append grid nodes to the raingauge node set
    # ------------------------------------------------------------------
    augmented = heterodata.clone()

    # Temporal features (sin/cos hour, sin/cos month, …) sit between the data
    # features and the LPE block.  They are identical for every node at a given
    # timestep, so grid nodes can copy them from the first real node.
    n_lpe_in_x = lpe_k if include_lpe else 0
    n_temporal  = F - _DATA_FEATURE_DIM - n_lpe_in_x  # 0 if no temporal enc.
    n_temporal  = max(n_temporal, 0)

    real_data = heterodata['raingauge'].x[:, :, :_DATA_FEATURE_DIM]       # [n_real, T, 2]
    grid_data = torch.zeros(n_grid, T, _DATA_FEATURE_DIM)                 # [n_grid, T, 2]

    if n_temporal > 0:
        # temporal slice: same value for all nodes → broadcast from node 0
        temporal = heterodata['raingauge'].x[0:1, :, _DATA_FEATURE_DIM:_DATA_FEATURE_DIM + n_temporal]
        real_combined = torch.cat([real_data, temporal.expand(n_real, -1, -1)], dim=2)
        grid_combined = torch.cat([grid_data, temporal.expand(n_grid, -1, -1)], dim=2)
    else:
        real_combined = real_data
        grid_combined = grid_data

    grid_y = torch.zeros(n_grid, T, 1)
    augmented['raingauge'].x = torch.cat([real_combined, grid_combined], dim=0)
    augmented['raingauge'].y = torch.cat([heterodata['raingauge'].y, grid_y], dim=0)

    grid_mask = torch.zeros(n_real + n_grid, dtype=torch.bool)
    grid_mask[n_real:] = True
    augmented['raingauge'].mask = grid_mask
    augmented['raingauge'].num_nodes = n_real + n_grid

    # ------------------------------------------------------------------
    # Step 2 – build grid-to-gauge KNN edges
    # ------------------------------------------------------------------
    gauge_latlon  = np.radians(mapping_df[['latitude', 'longitude']].values)
    grid_latlon   = np.radians(grid_coords[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=knn_gauge, metric='haversine')
    nbrs.fit(gauge_latlon)
    distances_rad, nbr_indices = nbrs.kneighbors(grid_latlon)

    earth_radius_km = 6371.0
    distances_km = distances_rad * earth_radius_km

    src, dst, weights = [], [], []
    for grid_i in range(n_grid):
        for k in range(knn_gauge):
            gauge_j = int(nbr_indices[grid_i, k])
            dist_km = float(distances_km[grid_i, k])
            src.append(n_real + grid_i)
            dst.append(gauge_j)
            weights.append(1.0 / max(dist_km, 1e-3))

    new_edge_index  = torch.tensor([src, dst], dtype=torch.long)
    new_edge_weight = torch.tensor(weights, dtype=torch.float32)
    new_edge_weight = new_edge_weight / new_edge_weight.max()

    rev_edge_index  = new_edge_index.flip(0)
    existing_idx    = augmented['raingauge', 'connects', 'raingauge'].edge_index
    existing_attr   = augmented['raingauge', 'connects', 'raingauge'].edge_attr
    augmented['raingauge', 'connects', 'raingauge'].edge_index = torch.cat(
        [existing_idx, new_edge_index, rev_edge_index], dim=1
    )
    augmented['raingauge', 'connects', 'raingauge'].edge_attr = torch.cat(
        [existing_attr, new_edge_weight, new_edge_weight], dim=0
    )

    # ------------------------------------------------------------------
    # Step 3 – recompute LPE for the augmented graph (optional)
    # ------------------------------------------------------------------
    if include_lpe and lpe_k > 0:
        temp = Data(
            x=torch.zeros(n_real + n_grid, 1),
            edge_index=augmented['raingauge', 'connects', 'raingauge'].edge_index,
            num_nodes=n_real + n_grid,
        )
        lpe_transform = AddLaplacianEigenvectorPE(k=lpe_k, attr_name='laplacian_pe')
        temp = lpe_transform(temp)
        lpe = temp.laplacian_pe                              # [N_total, lpe_k]
        lpe_expanded = lpe.unsqueeze(1).expand(-1, T, -1)   # [N_total, T, lpe_k]

        data_part = augmented['raingauge'].x[:, :, :_DATA_FEATURE_DIM + n_temporal]
        augmented['raingauge'].x = torch.cat([data_part, lpe_expanded], dim=2)

    # ------------------------------------------------------------------
    # Step 4 – run inference one timestep at a time
    # ------------------------------------------------------------------
    all_grid_preds: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(T):
            snap = HeteroData()

            snap['raingauge'].x         = augmented['raingauge'].x[:, t, :].to(device)
            snap['raingauge'].y         = augmented['raingauge'].y[:, t, :].to(device)
            snap['raingauge'].mask      = augmented['raingauge'].mask.to(device)
            snap['raingauge'].num_nodes = n_real + n_grid

            for node_type in augmented.node_types:
                if node_type == 'raingauge':
                    continue
                snap[node_type].x         = augmented[node_type].x[:, t, :].to(device)
                snap[node_type].num_nodes = augmented[node_type].x.shape[0]

            for edge_type in augmented.edge_types:
                snap[edge_type].edge_index = augmented[edge_type].edge_index.to(device)
                if hasattr(augmented[edge_type], 'edge_attr'):
                    snap[edge_type].edge_attr = augmented[edge_type].edge_attr.to(device)

            mask_t  = snap['raingauge'].mask
            x_input = snap['raingauge'].x.clone()
            x_input[mask_t, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {nt: snap[nt].x for nt in snap.node_types}
            x_dict['raingauge'] = x_input

            edge_attr_dict = {
                et: snap[et].edge_attr
                for et in snap.edge_types
                if hasattr(snap[et], 'edge_attr')
            }

            out = model(x_dict, snap.edge_index_dict, edge_attr_dict)

            grid_preds = out['raingauge'][mask_t].cpu().numpy().flatten()
            all_grid_preds.append(grid_preds)

    predictions = np.stack(all_grid_preds, axis=0)          # [T, n_grid]
    predictions = predictions.reshape(T, *grid_shape)       # [T, n_rows, n_cols]
    return predictions, grid_coords, grid_shape


# ---------------------------------------------------------------------------
# 3.  Visualisation helpers
# ---------------------------------------------------------------------------

def plot_rainfall_grid(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestamp_idx: int = 0,
    mapping_df: pd.DataFrame | None = None,
    title: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    show_outline: bool = True,
    log_scale: bool = False,
    ax=None,
):
    """
    Plot a single timestep of grid predictions as a filled spatial heatmap.

    Parameters
    ----------
    predictions   : np.ndarray  [T, n_rows, n_cols] from predict_on_grid
    grid_shape    : (n_rows, n_cols)
    bounds        : dict  {'left', 'right', 'top', 'bottom'}
    timestamp_idx : int   which timestep to show
    mapping_df    : DataFrame  optional – overlay real gauge positions as red dots
    title         : str   optional plot title
    vmin          : float colour-scale minimum (default 0.0)
    vmax          : float optional fixed colour-scale maximum (mm); auto if None
    boundaries    : list[float] optional explicit colour-transition breakpoints
    cmap          : str   matplotlib colourmap name (default 'YlGnBu')
    show_outline  : bool  overlay basemap (default True)
    log_scale     : bool  use log colour scale
    ax            : matplotlib Axes  optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    frame = predictions[timestamp_idx]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall grid. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  plot_rainfall_grid(..., **scales['rainfall'])"
        )

    import matplotlib.cm as cm_mod
    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else frame.max()
        norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm) [log scale]'
    elif boundaries is not None:
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm)'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm)'

    im = ax.imshow(
        frame,
        extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
        origin='lower',
        cmap=cmap_obj,
        norm=norm,
        aspect='auto',
        interpolation='bilinear',
    )
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    if mapping_df is not None:
        ax.scatter(
            mapping_df['longitude'], mapping_df['latitude'],
            c='red', s=25, zorder=5, label='Gauge stations',
            edgecolors='black', linewidths=0.4,
        )
        ax.legend(loc='lower right', fontsize=8)

    if show_outline:
        visualise_with_basemap(ax=ax)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or f'Predicted Rainfall — timestep {timestamp_idx}')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])
    return ax


def plot_rainfall_sequence(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestep_indices: list[int],
    mapping_df: pd.DataFrame | None = None,
    titles: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (5, 4),
    show_outline: bool = True,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    log_scale: bool = False,
    save_path: str | None = None,
):
    """
    Plot a row of N timestep snapshots in a single figure.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols]
    timestep_indices : list[int]   which timesteps to include (up to 6 recommended)
    titles           : list[str]   optional per-panel titles
    figsize_per_panel: (w, h)      size of each individual panel
    vmin             : float       shared colour-scale minimum (default 0.0)
    vmax             : float       shared colour-scale maximum; auto if None
    boundaries       : list[float] optional BoundaryNorm breakpoints (shared across panels)
    cmap             : str         matplotlib colourmap (default 'YlGnBu')
    save_path        : str         optional path to save the figure

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(timestep_indices)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    if n == 1:
        axes = [axes]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall sequence. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  plot_rainfall_sequence(..., **scales['rainfall'])"
        )

    for i, (t_idx, ax) in enumerate(zip(timestep_indices, axes)):
        panel_title = titles[i] if titles else f'Timestep {t_idx}'
        plot_rainfall_grid(
            predictions, grid_shape, bounds,
            timestamp_idx=t_idx,
            mapping_df=mapping_df,
            title=panel_title,
            vmin=vmin,
            vmax=vmax,
            boundaries=boundaries,
            cmap=cmap,
            show_outline=show_outline,
            log_scale=log_scale,
            ax=ax,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def animate_rainfall_grid(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestep_indices: list[int] | None = None,
    timestamps: list | None = None,
    mapping_df: pd.DataFrame | None = None,
    interval_ms: int = 200,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    show_outline: bool = True,
    log_scale: bool = False,
    figsize: tuple[float, float] = (9, 7),
    save_path: str | None = None,
) -> animation.FuncAnimation:
    """
    Animate grid predictions as a spatial heatmap time series.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols] from predict_on_grid
    timestep_indices : list[int]  subset of timesteps to animate; defaults to all
    timestamps       : list  optional display labels per timestep (e.g. pandas Timestamps)
    mapping_df       : DataFrame  optional – overlay real gauge positions as red dots
    interval_ms      : int  delay between frames in milliseconds
    vmin             : float  colour-scale minimum (default 0.0)
    vmax             : float  optional fixed colour-scale maximum (mm); auto if None
    boundaries       : list[float] optional BoundaryNorm breakpoints for discrete bands
    cmap             : str   matplotlib colourmap (default 'YlGnBu')
    figsize          : (w, h) figure size in inches
    save_path        : str  optional – save as .mp4 or .gif (requires ffmpeg / pillow)

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Call `HTML(anim.to_jshtml())` in a notebook cell to display inline.
    """
    import matplotlib.cm as cm_mod

    if timestep_indices is None:
        timestep_indices = list(range(predictions.shape[0]))

    subset = predictions[timestep_indices]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall animation. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  animate_rainfall_grid(..., **scales['rainfall'])"
        )

    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else subset.max()
        norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm) [log scale]'
    elif boundaries is not None:
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm)'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm)'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        subset[0],
        extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
        origin='lower',
        cmap=cmap_obj,
        norm=norm,
        aspect='auto',
        interpolation='bilinear',
        animated=True,
    )
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    if mapping_df is not None:
        ax.scatter(
            mapping_df['longitude'], mapping_df['latitude'],
            c='red', s=25, zorder=5, label='Gauge stations',
            edgecolors='black', linewidths=0.4,
        )
        ax.legend(loc='lower right', fontsize=8)

    if show_outline:
        visualise_with_basemap(ax=ax)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])

    label = (
        str(timestamps[0]) if timestamps else f'Timestep {timestep_indices[0]}'
    )
    title = ax.set_title(label, fontsize=11)

    def _update(frame_i):
        im.set_data(subset[frame_i])
        lbl = (
            str(timestamps[frame_i]) if timestamps
            else f'Timestep {timestep_indices[frame_i]}'
        )
        title.set_text(lbl)
        return im, title

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(timestep_indices),
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Animation saved to {save_path}")

    return anim


# ---------------------------------------------------------------------------
# 4.  ST model grid inference (GNNInductiveHeteroST)
# ---------------------------------------------------------------------------

def predict_on_grid_st(
    model,
    heterodata: HeteroData,
    mapping_df: pd.DataFrame,
    bounds: dict,
    window_size: int = 6,
    resolution_km: float = 1.0,
    knn_gauge: int = 5,
    device: str = 'cpu',
    include_lpe: bool = True,
    lpe_k: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int]]:
    """
    Predict rainfall on a regular grid for every valid timestep in *heterodata*
    using a spatio-temporal model (GNNInductiveHeteroST).

    Parameters
    ----------
    model        : trained GNNInductiveHeteroST (eval mode set internally)
    heterodata   : normalised HeteroData with [N, T, F] node features
    mapping_df   : DataFrame  columns ['longitude', 'latitude'] for real gauges
    bounds       : dict  {'left', 'right', 'top', 'bottom'}
    window_size  : int   number of preceding context timesteps (W)
    resolution_km: float grid spacing in km
    knn_gauge    : int   nearest real gauges to connect to each grid node
    device       : str   torch device string
    include_lpe  : bool  recompute Laplacian PE for the augmented graph
    lpe_k        : int   number of LPE eigenvectors (must match model training)

    Returns
    -------
    predictions  : np.ndarray  shape [T - window_size, n_rows, n_cols]
    grid_coords  : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape   : (n_rows, n_cols)
    """
    model.eval()

    grid_coords, grid_shape = generate_grid_coords(bounds, resolution_km)
    n_grid = len(grid_coords)
    n_real = heterodata['raingauge'].x.shape[0]
    T      = heterodata['raingauge'].x.shape[1]

    augmented = heterodata.clone()

    real_x = heterodata['raingauge'].x[:, :, :_DATA_FEATURE_DIM]
    grid_x = torch.zeros(n_grid, T, _DATA_FEATURE_DIM)
    grid_y = torch.zeros(n_grid, T, 1)
    augmented['raingauge'].x = torch.cat([real_x, grid_x], dim=0)
    augmented['raingauge'].y = torch.cat([heterodata['raingauge'].y, grid_y], dim=0)
    augmented['raingauge'].num_nodes = n_real + n_grid

    gauge_latlon = np.radians(mapping_df[['latitude', 'longitude']].values)
    grid_latlon  = np.radians(grid_coords[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=knn_gauge, metric='haversine')
    nbrs.fit(gauge_latlon)
    distances_rad, nbr_indices = nbrs.kneighbors(grid_latlon)
    distances_km = distances_rad * 6371.0

    src, dst, weights = [], [], []
    for grid_i in range(n_grid):
        for k in range(knn_gauge):
            gauge_j = int(nbr_indices[grid_i, k])
            dist_km = float(distances_km[grid_i, k])
            src.append(n_real + grid_i)
            dst.append(gauge_j)
            weights.append(1.0 / max(dist_km, 1e-3))

    new_edge_index  = torch.tensor([src, dst], dtype=torch.long)
    new_edge_weight = torch.tensor(weights, dtype=torch.float32)
    new_edge_weight = new_edge_weight / new_edge_weight.max()

    rev_edge_index = new_edge_index.flip(0)
    existing_idx   = augmented['raingauge', 'connects', 'raingauge'].edge_index
    existing_attr  = augmented['raingauge', 'connects', 'raingauge'].edge_attr
    augmented['raingauge', 'connects', 'raingauge'].edge_index = torch.cat(
        [existing_idx, new_edge_index, rev_edge_index], dim=1
    )
    augmented['raingauge', 'connects', 'raingauge'].edge_attr = torch.cat(
        [existing_attr, new_edge_weight, new_edge_weight], dim=0
    )

    if include_lpe and lpe_k > 0:
        temp = Data(
            x=torch.zeros(n_real + n_grid, 1),
            edge_index=augmented['raingauge', 'connects', 'raingauge'].edge_index,
            num_nodes=n_real + n_grid,
        )
        lpe_transform = AddLaplacianEigenvectorPE(k=lpe_k, attr_name='laplacian_pe')
        temp = lpe_transform(temp)
        lpe          = temp.laplacian_pe
        lpe_expanded = lpe.unsqueeze(1).expand(-1, T, -1)

        data_part = augmented['raingauge'].x[:, :, :_DATA_FEATURE_DIM]
        augmented['raingauge'].x = torch.cat([data_part, lpe_expanded], dim=2)

    edge_index_dict = {
        et: augmented[et].edge_index.to(device)
        for et in augmented.edge_types
    }
    edge_attr_dict = {
        et: augmented[et].edge_attr.to(device)
        for et in augmented.edge_types
        if hasattr(augmented[et], 'edge_attr')
    }

    all_grid_preds: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(window_size, T):
            ctx_idx = list(range(t - window_size, t))

            x_dict: dict = {}
            x_context_dict: dict = {}

            x_cur = augmented['raingauge'].x[:, t, :].clone().to(device)
            x_cur[n_real:, :_DATA_FEATURE_DIM] = 0.0
            x_dict['raingauge']         = x_cur
            x_context_dict['raingauge'] = augmented['raingauge'].x[:, ctx_idx, :].to(device)

            for ntype in augmented.node_types:
                if ntype == 'raingauge':
                    continue
                x_dict[ntype]         = augmented[ntype].x[:, t, :].to(device)
                x_context_dict[ntype] = augmented[ntype].x[:, ctx_idx, :].to(device)

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            grid_preds = out['raingauge'][n_real:].cpu().numpy().flatten()
            all_grid_preds.append(grid_preds)

    predictions = np.stack(all_grid_preds, axis=0)
    predictions = predictions.reshape(len(all_grid_preds), *grid_shape)
    return predictions, grid_coords, grid_shape


# ---------------------------------------------------------------------------
# 5.  Multi-source comparison grid  (Figure 14 style)
# ---------------------------------------------------------------------------

def plot_source_comparison(
    sources: list[dict],
    bounds: dict,
    timestep_indices: list[int],
    timestamps: list | None = None,
    figsize_per_panel: tuple[float, float] = (4, 3.5),
    show_outline: bool = True,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """
    Multi-row × multi-column comparison figure.
    Rows = data sources / model variants.  Columns = timesteps.

    Parameters
    ----------
    sources : list of dicts, one per row.  Each dict may contain:
        'label'      : str            row label shown as y-axis title on the
                                      leftmost panel
        'data'       : np.ndarray     [T, n_rows, n_cols]
        'vmin'       : float          colour-scale minimum (default 0.0)
        'vmax'       : float          colour-scale maximum  ← required unless
                                      'boundaries' is set
        'boundaries' : list[float]    optional BoundaryNorm breakpoints
        'cmap'       : str            colourmap (default 'YlGnBu')
        'cbar_label' : str            colourbar label
        'mapping_df' : pd.DataFrame   optional gauge-dot overlay (per row)
        'log_scale'  : bool           log colour scale (default False)
    bounds            : dict  {'left', 'right', 'top', 'bottom'}
    timestep_indices  : list[int]  which timesteps (into each source's array)
                        to show as columns
    timestamps        : list  optional human-readable column header labels
                        (e.g. list of pd.Timestamps).  Falls back to
                        "t=<idx>" if None.
    figsize_per_panel : (w, h) in inches for each individual panel
    show_outline      : bool  overlay CARTO basemap on every panel
    suptitle          : str   optional overall figure title
    save_path         : str   optional path to save the figure

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    sources = [
        {'label': 'Radar',              'data': radar_grid,
         'vmin': 0, 'vmax': 20,         'cmap': 'Blues',
         'cbar_label': 'Reflectivity'},
        {'label': 'IDW',                'data': idw_grid,
         'vmin': 0, 'vmax': 50,
         'boundaries': [0,0.5,1,2,5,10,20,40,80],
         'mapping_df': mapping_df},
        {'label': 'Gauge GNN',          'data': gauge_grid,
         'vmin': 0, 'vmax': 50,
         'boundaries': [0,0.5,1,2,5,10,20,40,80]},
        {'label': 'Gauge+Radar GNN',    'data': radar_gnn_grid,
         'vmin': 0, 'vmax': 50,
         'boundaries': [0,0.5,1,2,5,10,20,40,80]},
        {'label': 'Gauge+Radar+Sat GNN','data': sat_gnn_grid,
         'vmin': 0, 'vmax': 50,
         'boundaries': [0,0.5,1,2,5,10,20,40,80]},
    ]
    plot_source_comparison(sources, bounds, timestep_indices=[10, 20, 30],
                           timestamps=my_timestamps[[10, 20, 30]])
    """
    import matplotlib.cm as cm_mod

    n_rows = len(sources)
    n_cols = len(timestep_indices)
    w, h   = figsize_per_panel

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w * n_cols, h * n_rows),
        squeeze=False,
    )

    for row_i, src in enumerate(sources):
        data       = src['data']                          # [T, H, W]
        vmin       = src.get('vmin', 0.0)
        vmax       = src.get('vmax', None)
        boundaries = src.get('boundaries', None)
        cmap_name  = src.get('cmap', 'YlGnBu')
        cbar_label = src.get('cbar_label', 'Predicted Rainfall (mm/hr)')
        mapping_df = src.get('mapping_df', None)
        log_scale  = src.get('log_scale', False)
        label      = src.get('label', f'Source {row_i}')

        is_last_row = (row_i == n_rows - 1)

        if vmax is None and boundaries is None:
            raise ValueError(
                f"Source '{label}': either 'vmax' or 'boundaries' must be set."
            )

        cmap_obj = cm_mod.get_cmap(cmap_name)
        if log_scale:
            _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
            _lv_max = vmax if vmax else data[timestep_indices].max()
            norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        elif boundaries is not None:
            norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for col_i, t_idx in enumerate(timestep_indices):
            ax    = axes[row_i, col_i]
            frame = data[t_idx]

            is_last_col = (col_i == n_cols - 1)

            # 1) Rainfall first
            im = ax.imshow(
                frame,
                extent=[bounds['left'], bounds['right'],
                        bounds['bottom'], bounds['top']],
                origin='lower',
                cmap=cmap_obj,
                norm=norm,
                aspect='auto',
                interpolation='bilinear',
                zorder=1,
            )

            if mapping_df is not None:
                ax.scatter(
                    mapping_df['longitude'], mapping_df['latitude'],
                    c='red', s=15, zorder=5,
                    edgecolors='black', linewidths=0.3,
                )

            # 2) Set limits BEFORE basemap so tiles cover the right extent
            ax.set_xlim(bounds['left'],  bounds['right'])
            ax.set_ylim(bounds['bottom'], bounds['top'])

            # 3) Basemap on top as a semi-transparent overlay
            if show_outline:
                import contextily as ctx
                ctx.add_basemap(
                    ax, crs=4326,
                    source=ctx.providers.CartoDB.PositronNoLabels,
                    alpha=0.45,
                    zorder=2,
                )
                # Remove attribution from every panel except the bottom-right
                if not (is_last_row and is_last_col):
                    for txt in ax.texts:
                        if 'OpenStreetMap' in txt.get_text() or 'CARTO' in txt.get_text():
                            txt.set_visible(False)

            # Darker, thicker border on every panel
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)

            # Column header: only on the top row
            if row_i == 0:
                col_title = (
                    str(timestamps[col_i]) if timestamps is not None
                    else f't={t_idx}'
                )
                ax.set_title(col_title, fontsize=11, pad=5)

            # Row label: only on the leftmost column
            if col_i == 0:
                ax.set_ylabel(label, fontsize=13, fontweight='bold', labelpad=8)
            else:
                ax.set_ylabel('')

            # Latitude ticks: leftmost column only
            if col_i == 0:
                ax.tick_params(axis='y', labelsize=8)
            else:
                ax.tick_params(axis='y', left=False, labelleft=False)

            # Longitude ticks: bottom row only
            if is_last_row:
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlabel('Longitude', fontsize=9)
            else:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
                ax.set_xlabel('')

        # One colourbar per row, attached to the rightmost panel
        cbar = fig.colorbar(
            im, ax=axes[row_i, -1],
            label=cbar_label, shrink=0.85, pad=0.02,
        )
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
