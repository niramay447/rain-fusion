import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData


class AURadarGraph:
    """
    Wraps BOM Rainfields3 radar data as a HeteroData layer for use with
    GaugeGraphNew.add_heterodata().

    Parameters
    ----------
    radar_df   : DataFrame with columns 'timestamp' (tz-aware UTC) and
                 'data' (np.ndarray of shape (n_nodes,)) — output of
                 AURadarPreprocessor.build_dataset / build_dataset_range
    gauge_index: DatetimeIndex from the gauge DataFrame; radar features are
                 aligned to these timestamps (missing → zero)
    node_coords: DataFrame with 'latitude' and 'longitude' columns — output
                 of AURadarPreprocessor.get_node_coords()
    grid_shape : (H, W) tuple — output of AURadarPreprocessor.grid_shape,
                 used to build 8-connectivity radar-radar edges
    """

    def __init__(
        self,
        radar_df: pd.DataFrame,
        gauge_index: pd.DatetimeIndex,
        node_coords: pd.DataFrame,
        grid_shape: tuple[int, int],
    ):
        self.grid_coords = node_coords
        self.grid_shape = grid_shape
        self.n_nodes = len(node_coords)

        # Build timestamp → array lookup; normalise to UTC-aware for safe comparison
        radar_lookup: dict = {}
        for row in radar_df.itertuples(index=False):
            ts = row.timestamp
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            radar_lookup[ts] = row.data  # shape (n_nodes,)

        # Normalise gauge_index to UTC-aware
        if gauge_index.tzinfo is None:
            gauge_index_utc = gauge_index.tz_localize(
                "Australia/Sydney", ambiguous="NaT", nonexistent="shift_forward"
            ).tz_convert("UTC")
        else:
            gauge_index_utc = gauge_index.tz_convert("UTC")

        T = len(gauge_index_utc)
        # 2 channels: [accum_mm, valid_flag]
        # valid_flag = 1 where a real radar scan exists for this timestamp,
        # 0 where the timestep is zero-filled because radar had no data. Without
        # this, a zero-filled gap is indistinguishable from an observed zero
        # (matches the satellite layer's valid-flag convention).
        feature_matrix = np.zeros((self.n_nodes, T, 2), dtype=np.float32)
        for t, ts in enumerate(gauge_index_utc):
            arr = radar_lookup.get(ts)
            if arr is not None:
                feature_matrix[:, t, 0] = arr
                feature_matrix[:, t, 1] = 1.0
            # else stays [0.0, 0.0]

        # (n_nodes, T, 2)
        x = torch.tensor(feature_matrix, dtype=torch.float32)

        self._heterodata = HeteroData()
        self._heterodata["radar"].x = x

        # 8-connectivity grid edges (undirected — both directions)
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
        self._heterodata["radar", "neighbors", "radar"].edge_index = edge_index
        self._heterodata["radar", "neighbors", "radar"].edge_attr = torch.ones(
            edge_index.shape[1], dtype=torch.float32
        )

        # Self-loops so h_dict['radar'] survives every GNN layer even when
        # there are no incoming cross-type edges in early message-passing rounds
        self_idx = torch.arange(self.n_nodes, dtype=torch.long)
        self_edge = torch.stack([self_idx, self_idx], dim=0)
        self._heterodata["radar", "self", "radar"].edge_index = self_edge
        self._heterodata["radar", "self", "radar"].edge_attr = torch.ones(
            self.n_nodes, dtype=torch.float32
        )

    def get_radar_heterodata(self) -> HeteroData:
        return self._heterodata
