import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch.linalg import vector_norm
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GINEConv, to_hetero, HeteroConv, GCNConv, GATConv, Linear, GraphConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )


        for _ in range(num_layers):
            conv = HeteroConv({
                ('general_station', 'gen_to_gen', 'general_station'):
                    GraphConv((-1, -1), hidden_channels),
                ('general_station', 'gen_to_rain', 'rainfall_station'):
                    GraphConv((-1,-1), hidden_channels),
                ('rainfall_station', 'rain_to_gen', 'general_station'):
                    GraphConv((-1,-1), hidden_channels),
                ('rainfall_station', 'rain_to_rain', 'rainfall_station'):
                    GraphConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)

        # Add layer normalization
        self.norm_general = torch.nn.LayerNorm(hidden_channels)
        self.norm_rainfall = torch.nn.LayerNorm(hidden_channels)

        # torch.nn.init.xavier_uniform_(self.lin_rainfall.weight)
        # torch.nn.init.xavier_uniform_(self.lin_general.weight)
        # torch.nn.init.constant_(self.lin_rainfall.bias, 1.0)
        # torch.nn.init.constant_(self.lin_general.bias, 1.0)

    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # Initialize with zeros so first layer only gets neighbor info
        h_dict = {key: torch.zeros_like(x) for key, x in x_dict.items()}

        # First layer: aggregate from original features
        for conv in self.convs:
            h_dict = conv({key: x for key, x in x_dict.items()},
                            edge_index_dict, edge_attributes_dict)
            # # Use Leaky ReLU in hidden layers
            # h_dict = {key: F.leaky_relu(x, negative_slope=0.01) for key, x in h_dict.items()}
            h_dict = {key: x.relu() for key, x in h_dict.items()}

        # # Normalize before output layer
        # h_gen_norm = self.norm_general(h_dict['general_station'])
        # h_rain_norm = self.norm_rainfall(h_dict['rainfall_station'])

        # return {
        #     'general_station': F.softplus(self.lin_general(h_gen_norm)),
        #     'rainfall_station': F.softplus(self.lin_rainfall(h_rain_norm))
        # }

        return {
            'general_station': F.relu(self.lin_general(h_dict['general_station'])),
            'rainfall_station': F.relu(self.lin_rainfall(h_dict['rainfall_station']))
        }

class HeteroGNN2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )

        # for _ in range(num_layers - 1):
        #     conv = HeteroConv({
        #         ('general_station', 'gen_to_gen', 'general_station'):
        #             GraphConv((-1, -1), hidden_channels),
        #         ('general_station', 'gen_to_rain', 'rainfall_station'):
        #             GraphConv((-1, -1), hidden_channels),
        #         ('rainfall_station', 'rain_to_gen', 'general_station'):
        #             GraphConv((-1, -1), hidden_channels),
        #         ('rainfall_station', 'rain_to_rain', 'rainfall_station'):
        #             GraphConv((-1, -1), hidden_channels),
        #     }, aggr='mean')
        #     self.convs.append(conv)
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'):
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('general_station', 'gen_to_rain', 'rainfall_station'):
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_gen', 'general_station'):
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'):
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()},
                            edge_index_dict, edge_attributes_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}


        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }

class HeteroGCNGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'):
                        GCNConv(-1, hidden_channels, add_self_loops=True),
                    ('general_station', 'gen_to_rain', 'rainfall_station'):
                        GCNConv(-1, hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_gen', 'general_station'):
                        GCNConv(-1, hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'):
                        GCNConv(-1, hidden_channels, add_self_loops=True),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, edge_weights_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()},
                            edge_index_dict, edge_weights_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}


        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }



class HeteroSAGEGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'):
                        SAGEConv(-1, hidden_channels),
                    ('general_station', 'gen_to_rain', 'rainfall_station'):
                        SAGEConv(-1, hidden_channels),
                    ('rainfall_station', 'rain_to_gen', 'general_station'):
                        SAGEConv(-1, hidden_channels),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'):
                        SAGEConv(-1, hidden_channels),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()},
                            edge_index_dict, edge_attributes_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}


        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )


        for _ in range(num_layers):
            conv = GraphConv((-1, -1), hidden_channels)
            self.convs.append(conv)

        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)

        # Add layer normalization
        self.norm_general = torch.nn.LayerNorm(hidden_channels)
        self.norm_rainfall = torch.nn.LayerNorm(hidden_channels)

        # torch.nn.init.xavier_uniform_(self.lin_rainfall.weight)
        # torch.nn.init.xavier_uniform_(self.lin_general.weight)
        # torch.nn.init.constant_(self.lin_rainfall.bias, 1.0)
        # torch.nn.init.constant_(self.lin_general.bias, 1.0)

    def forward(self, x, edge_index, edge_attributes):
        if x.dim() != 2:
            raise ValueError(f"GNN.forward expects x with shape [N, F], got {x.shape}")

        # First layer: aggregate from original features
        for conv in self.convs:
            x = conv(x, edge_index, edge_attributes)
            x = x.relu()


        # # Normalize before output layer
        # h_gen_norm = self.norm_general(h_dict['general_station'])
        # h_rain_norm = self.norm_rainfall(h_dict['rainfall_station'])
        return F.relu(self.lin_general(x))

class GNNInductive(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))

        self.lin_general = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attributes=None):
        """
        IMPORTANT: GraphConv only takes (x, edge_index, edge_weight)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attributes)

            x = F.relu(x)

        out = self.lin_general(x)

        return out

class GNNInductiveHetero(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, num_layers, edge_types, dropout=0.0, hetero_aggr='mean'):
        """
        Args:
            in_channels_dict: dict
                {
                    'raingauge': F_g,
                    'radar': F_r
                }
            dropout: dropout probability applied after each GNN layer's activation.
                     0.0 disables dropout entirely.
            hetero_aggr: how a node combines messages ACROSS edge types.
                     'mean' (default) gives every source equal 1/n_types weight and
                     rescales the useful signal by 1/n_types — a weak source
                     (satellite) then dilutes gauge+radar and can tip training into
                     the predict-zero basin. 'sum' lets the model drive a weak
                     source toward 0 but changes magnitude scale. 'gate' learns a
                     softmax weight per source (per dst node type, per layer) so the
                     model down-weights redundant/weak supports automatically —
                     fixes both Sydney dilution and Wagga collapse. Gates init to
                     favor the same-support relation (~0.8 self) so training starts
                     healthy and *learns* to add cross-support 2D sources rather
                     than collapsing on a weak one. Readable via gate_weights().
        """
        super().__init__()

        self.config = dict(
            in_channels=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.use_gate = (hetero_aggr == 'gate' and edge_types is not None)

        if self.use_gate:
            # Learned per-source GATING: replace the fixed mean/sum combine across
            # edge types with a softmax-weighted combination, so the model can
            # down-weight a redundant/weak source (radar/satellite) toward 0
            # instead of being forced to average it in at 1/n_types. One GraphConv
            # per edge type per layer; one gate logit-vector per (dst node type,
            # layer). Gates init at 0 → softmax = uniform = exactly 'mean' at start.
            self.edge_types = list(edge_types)
            self.dst_to_ets = {}                       # dst node type -> [edge types]
            for et in self.edge_types:
                self.dst_to_ets.setdefault(et[2], []).append(et)
            self.gate_convs = torch.nn.ModuleList()
            self.gate_logits = torch.nn.ModuleList()
            for _ in range(num_layers):
                cdict = torch.nn.ModuleDict()
                for et in self.edge_types:
                    cdict[self._ekey(et)] = GraphConv((-1, -1), hidden_channels)
                self.gate_convs.append(cdict)
                ldict = torch.nn.ParameterDict()
                for dst, ets in self.dst_to_ets.items():
                    # Init gates to FAVOR the same-support relation (src==dst, e.g.
                    # gauge-gauge) and start cross-support (2D→0D) contributions low.
                    # The model thus begins ≈ same-support-only (healthy) and *learns*
                    # to bring in radar/satellite — instead of being forced to take a
                    # weak source at 1/n from step 0 and diving into the predict-zero
                    # basin before the gate can adapt. Cross-support is learned, not
                    # suppressed: gates are free to rise if a source helps.
                    init = torch.zeros(len(ets))
                    for j, et in enumerate(ets):
                        if et[0] == et[2]:          # self-support relation
                            init[j] = 2.0           # softmax → ~0.8 self at start
                    ldict[dst] = torch.nn.Parameter(init)
                self.gate_logits.append(ldict)
        else:
            self.convs = torch.nn.ModuleList()
            for layer_idx in range(num_layers):
                if edge_types is None:
                    conv = HeteroConv({
                        ('raingauge', 'connects', 'raingauge'):
                            GraphConv((-1, -1), hidden_channels),
                    }, aggr='sum')
                else:
                    conv_dict = {}
                    for edge_type in edge_types:
                        conv_dict[edge_type] = GraphConv((-1, -1), hidden_channels)
                    conv = HeteroConv(conv_dict, aggr=hetero_aggr)
                self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    @staticmethod
    def _ekey(edge_type):
        """ModuleDict-safe string key for an edge-type tuple."""
        return "__".join(edge_type)

    @torch.no_grad()
    def gate_weights(self):
        """
        Learned per-source softmax weights, for the interpretability figure.
        Returns {dst_node_type: {edge_type: [w_layer0, w_layer1, ...]}}.
        Only meaningful when hetero_aggr='gate'. Reads how much each support
        contributes (e.g. radar≈0.6, satellite≈0.05 → the model learned to
        discount the weak source).
        """
        if not getattr(self, "use_gate", False):
            return {}
        out = {dst: {et: [] for et in ets} for dst, ets in self.dst_to_ets.items()}
        for layer in range(len(self.gate_logits)):
            for dst, ets in self.dst_to_ets.items():
                w = torch.softmax(self.gate_logits[layer][dst], dim=0).tolist()
                for et, wi in zip(ets, w):
                    out[dst][et].append(wi)
        return out

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        if self.use_gate:
            h = dict(x_dict)
            for layer in range(len(self.gate_convs)):
                cdict, ldict = self.gate_convs[layer], self.gate_logits[layer]
                msgs = {dst: [] for dst in self.dst_to_ets}
                for et in self.edge_types:
                    src, _, dst = et
                    ew = edge_attr_dict.get(et) if edge_attr_dict is not None else None
                    msgs[dst].append(
                        cdict[self._ekey(et)]((h[src], h[dst]),
                                              edge_index_dict[et], ew)
                    )
                new_h = {}
                for dst, outs in msgs.items():
                    if len(outs) == 1:
                        new_h[dst] = outs[0]
                    else:
                        w = torch.softmax(ldict[dst], dim=0)   # learned source weights
                        new_h[dst] = sum(w[i] * outs[i] for i in range(len(outs)))
                h = {**h, **new_h}
                h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
            return {'raingauge': self.lin(h['raingauge'])}

        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(
                h_dict,
                edge_index_dict,
                edge_weight_dict=edge_attr_dict,
            )
            h_dict = {k: self.dropout(F.relu(v)) for k, v in h_dict.items()}

        # No output activation: allow any real value during training so that
        # MSE gradients never vanish (softplus saturates for large-negative
        # linear outputs, killing gradients during dry periods).
        # Clamp to >= 0 happens in test_model / validate at evaluation time.
        out_dict = {
            'raingauge': self.lin(h_dict['raingauge']),
        }

        return out_dict
