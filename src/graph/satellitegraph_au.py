import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData


class AUSatelliteGraph:
    """
    Wraps Himawari-8/9 CRRPH satellite data as a HeteroData layer for use
    with GaugeGraphNew.add_heterodata().

    Parameters
    ----------
    satellite_df : DataFrame with columns 'timestamp' (tz-aware UTC) and
                   'data' (np.ndarray of shape (n_nodes,)) — output of
                   AUSatellitePreprocessor.build_dataset / build_dataset_range
    gauge_index  : DatetimeIndex from the gauge DataFrame; satellite features
                   are aligned to these timestamps (missing → zero)
    node_coords  : DataFrame with 'latitude' and 'longitude' columns — output
                   of AUSatellitePreprocessor.get_node_coords()
    grid_shape   : (H, W) tuple — output of AUSatellitePreprocessor.grid_shape,
                   used to build 8-connectivity satellite-satellite edges
    """

    def __init__(
        self,
        satellite_df: pd.DataFrame,
        gauge_index: pd.DatetimeIndex,
        node_coords: pd.DataFrame,
        grid_shape: tuple[int, int],
    ):
        self.grid_coords = node_coords
        self.grid_shape  = grid_shape
        self.n_nodes     = len(node_coords)

        # Build timestamp → array lookup; normalise to UTC-aware
        satellite_lookup: dict = {}
        for row in satellite_df.itertuples(index=False):
            ts = row.timestamp
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            satellite_lookup[ts] = row.data  # shape (n_nodes,)

        zero = np.zeros((self.n_nodes, 2), dtype=np.float32)  # [accum, valid_flag]

        if gauge_index.tzinfo is None:
            gauge_index_utc = gauge_index.tz_localize(
                "Australia/Sydney", ambiguous="NaT", nonexistent="shift_forward"
            ).tz_convert("UTC")
        else:
            gauge_index_utc = gauge_index.tz_convert("UTC")

        T = len(gauge_index_utc)
        # 2 channels: [accum_mm, valid_flag]
        feature_matrix = np.zeros((self.n_nodes, T, 2), dtype=np.float32)
        for t, ts in enumerate(gauge_index_utc):
            arr = satellite_lookup.get(ts, zero)  # (n_nodes, 2)
            feature_matrix[:, t, :] = arr

        # Raw accumulation on the rain channel — no log transform. Scale is
        # handled by per-feature z-score normalisation (compute_norm_stats),
        # matching the no-log convention used for every source.
        # (n_nodes, T, 2)
        x = torch.tensor(feature_matrix, dtype=torch.float32)

        self._heterodata = HeteroData()
        self._heterodata["satellite"].x = x

        # 8-connectivity grid edges (undirected — both directions stored)
        H, W = grid_shape
        src, dst = [], []
        for r in range(H):
            for c in range(W):
                node = r * W + c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            nb = nr * W + nc
                            src.append(node)
                            dst.append(nb)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        self._heterodata["satellite", "neighbors", "satellite"].edge_index = edge_index
        self._heterodata["satellite", "neighbors", "satellite"].edge_attr = torch.ones(
            edge_index.shape[1], dtype=torch.float32
        )

        # Self-loops so satellite embeddings survive every GNN layer
        self_idx  = torch.arange(self.n_nodes, dtype=torch.long)
        self_edge = torch.stack([self_idx, self_idx], dim=0)
        self._heterodata["satellite", "self", "satellite"].edge_index = self_edge
        self._heterodata["satellite", "self", "satellite"].edge_attr = torch.ones(
            self.n_nodes, dtype=torch.float32
        )

    def get_satellite_heterodata(self) -> HeteroData:
        return self._heterodata
