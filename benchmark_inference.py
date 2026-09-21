"""
Standalone inference-cost benchmark for the rebuttal's scalability response
(Reviewer 3 Major Comment 6 + Meta Comment 2).

Reuses the exact graph-building / model-construction code path from train.py
(fold 0 only, no training) so the reported numbers match what's actually
deployed. Run on the SAME hardware class used for the training-time numbers
already in logs/wagga_gauge_hp_*.log, so both figures are apples-to-apples.

Reports:
  - graph size: node/edge counts per type
  - model size: total trainable params
  - inference latency: mean/std wall-clock per forward pass, both per-batch
    and normalised to a single hourly snapshot (one graph instance)
  - hardware: device name, CUDA availability

Usage:
    python benchmark_inference.py --config config_gauge_radar_satellite_notemporal.yaml \
        --checkpoint experiments/20260605_165424_gauge_radar_satellite_notemporal_gate/weather_gnn_best_0.pth \
        --n-repeats 50
"""
from src.sampling.main import stratified_spatial_kfold_dual  # noqa: F401  (import-order fix, see train.py)

import argparse
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_gauge_radar_satellite_notemporal.yaml')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--n-repeats', type=int, default=50,
                     help='Forward passes to time per batch (after warmup) for a stable mean/std.')
parser.add_argument('--n-warmup', type=int, default=5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_config(args.config)
batch_size = config['training_params']['batch_size']

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

from src.sampling.main import stratified_spatial_kfold_dual as _kfold
split_info = _kfold(raingauge_station_mappings_df, seed=123, plot=False,
                     n_splits=config['training_params']['fold_count'])

datasources = config.get('datasources', ['raingauge'])
use_radar = 'radar' in datasources
use_satellite = 'satellite' in datasources

radar_graph = None
if use_radar:
    from src.graph.radargraph_au import AURadarGraph
    import pickle
    radar_cfg = config['dataset_parameters']
    radar_site = radar_cfg.get('radar_site', 'sydney')
    if radar_site == 'wagga':
        from src.radar.wagga_radar_preprocessor import AURadarPreprocessor
    else:
        from src.radar.au_preprocessor import AURadarPreprocessor
    with open(radar_cfg['radar_cache_path'], 'rb') as f:
        radar_df_raw = pickle.load(f)
    preprocessor = AURadarPreprocessor(
        radar_cfg.get('radar_base_path', '/g/data/rq0/rainfields3'),
        radar_cfg.get('radar_id', 71),
        radar_cfg.get('radar_stride', 10),
    )
    radar_graph = AURadarGraph(
        radar_df=radar_df_raw,
        gauge_index=raingauge_df.index,
        node_coords=preprocessor.get_node_coords(),
        grid_shape=preprocessor.grid_shape,
    )

satellite_graph = None
if use_satellite:
    from src.graph.satellitegraph_au import AUSatelliteGraph
    import pickle
    sat_cfg = config['dataset_parameters']
    sat_site = sat_cfg.get('satellite_site', 'sydney')
    if sat_site == 'wagga':
        from src.satellite.wagga_satellite_preprocessor import AUSatellitePreprocessor
    else:
        from src.satellite.au_preprocessor import AUSatellitePreprocessor
    with open(sat_cfg['satellite_cache_path'], 'rb') as f:
        satellite_df_raw = pickle.load(f)
    preprocessor_sat = AUSatellitePreprocessor(
        sat_cfg.get('satellite_base_path',
                     '/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph'))
    satellite_graph = AUSatelliteGraph(
        satellite_df=satellite_df_raw,
        gauge_index=raingauge_df.index,
        node_coords=preprocessor_sat.get_node_coords(),
        grid_shape=preprocessor_sat.grid_shape,
    )

gauge_graph = GaugeGraphNew(
    raingauge_df, raingauge_station_mappings_df,
    split_info=split_info[args.fold],
    knn=config['layer_connect']['gauge_gauge'], config=config,
)
if use_radar:
    gauge_graph.add_heterodata(
        heterodata_layer=radar_graph.get_radar_heterodata(),
        coords=radar_graph.grid_coords, layer_name='radar',
        knn=config['layer_connect']['radar_gauge'],
    )
if use_satellite:
    gauge_graph.add_heterodata(
        heterodata_layer=satellite_graph.get_satellite_heterodata(),
        coords=satellite_graph.grid_coords, layer_name='satellite',
        knn=config['layer_connect']['satellite_gauge'],
    )

test_data = gauge_graph.get_test_heterodata()

print("\n=== Graph size (single hourly snapshot) ===")
total_nodes, total_edges = 0, 0
for ntype in test_data.node_types:
    n = test_data[ntype].num_nodes
    total_nodes += n
    print(f"  {ntype:12s} nodes: {n}")
for etype in test_data.edge_types:
    e = test_data[etype].edge_index.shape[1]
    total_edges += e
    print(f"  {etype} edges: {e}")
print(f"  TOTAL nodes: {total_nodes}, TOTAL edges: {total_edges}")

hidden_channels = config['model']['hidden_channels']
num_layers = config['model']['num_layers']
use_bg = config['training_params'].get('use_bernoulli_gamma', False)
out_channels = 3 if use_bg else 1
in_channels_dict = {"raingauge": 1}
if use_radar:
    in_channels_dict["radar"] = 2
if use_satellite:
    in_channels_dict["satellite"] = 2

model = GNNInductiveHetero(
    in_channels_dict=in_channels_dict,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_layers=num_layers,
    edge_types=test_data.edge_types,
    dropout=config['model']['dropout'],
    hetero_aggr=config['model'].get('hetero_aggr', 'mean'),
).to(device)

# Lazy modules (PyG "-1" in_channels) only materialise their weights after
# one forward pass, so run a dummy pass BEFORE loading the checkpoint.
test_loader = GeometricDataLoader(
    HeterogeneousWeatherGraphDatasetInductive(test_data),
    batch_size=batch_size, shuffle=False,
)
_DATA_FEATURE_DIM = 2

sample_batch = next(iter(test_loader)).to(device)
with torch.no_grad():
    x_dict = {nt: sample_batch[nt].x for nt in sample_batch.node_types}
    edge_attr_dict = {
        et: sample_batch[et].edge_attr for et in sample_batch.edge_types
        if hasattr(sample_batch[et], 'edge_attr')
    }
    model(x_dict, sample_batch.edge_index_dict, edge_attr_dict)

state_dict = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"\n=== Model size ===\n  Total trainable params: {total_params:,}")

print(f"\n=== Hardware ===")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------------
# Timed inference: one batch = num_graphs stacked hourly snapshots.
# Mirrors test_model()'s masking exactly (single vectorized zero, ONE
# forward pass for the whole batch — not a per-node leave-one-out loop).
# ------------------------------------------------------------------------
latencies_ms = []
with torch.no_grad():
    it = iter(test_loader)
    for _ in range(args.n_warmup):
        try:
            batch = next(it).to(device)
        except StopIteration:
            it = iter(test_loader)
            batch = next(it).to(device)
        x = batch['raingauge'].x
        mask = batch['raingauge'].mask
        x_masked = x.clone()
        x_masked[mask, :_DATA_FEATURE_DIM] = 0.0
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        x_dict['raingauge'] = x_masked
        edge_attr_dict = {
            et: batch[et].edge_attr for et in batch.edge_types
            if hasattr(batch[et], 'edge_attr')
        }
        model(x_dict, batch.edge_index_dict, edge_attr_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    num_graphs_per_batch = None
    for _ in range(args.n_repeats):
        try:
            batch = next(it).to(device)
        except StopIteration:
            it = iter(test_loader)
            batch = next(it).to(device)
        x = batch['raingauge'].x
        mask = batch['raingauge'].mask
        num_graphs_per_batch = batch['raingauge'].ptr.size(0) - 1
        x_masked = x.clone()
        x_masked[mask, :_DATA_FEATURE_DIM] = 0.0
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        x_dict['raingauge'] = x_masked
        edge_attr_dict = {
            et: batch[et].edge_attr for et in batch.edge_types
            if hasattr(batch[et], 'edge_attr')
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x_dict, batch.edge_index_dict, edge_attr_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

latencies_ms = np.array(latencies_ms)
per_snapshot_ms = latencies_ms / num_graphs_per_batch

print(f"\n=== Inference latency (n={args.n_repeats} forward passes, "
      f"batch_size={batch_size} hourly snapshots) ===")
print(f"  Per batch:    {latencies_ms.mean():.2f} ± {latencies_ms.std():.2f} ms")
print(f"  Per snapshot: {per_snapshot_ms.mean():.3f} ± {per_snapshot_ms.std():.3f} ms "
      f"(i.e. one full-network prediction, all held-out nodes, single forward pass)")
