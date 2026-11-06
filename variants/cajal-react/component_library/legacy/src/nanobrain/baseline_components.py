"""Composable upstream neural components; adapters are not whole-method replicas.

Only input/output alignment lives here. Implementations are imported from the
checked-in reference sources or installed libraries, never copied/reimplemented.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = (
    "reference_methods/scBrain/src/models/mjm.py",
    "reference_methods/scBrain/src/models/spatial_gnn.py",
    "reference_methods/scFiT/src/scfit/model.py",
    "reference_methods/scFiT/src/scfit/utils.py",
    "reference_methods/scFiT/third_party/scNODE-main/model/layer.py",
    "reference_methods/scUniFlow/src/scuniflow/model.py",
    *tuple(str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / "reference_methods/TrajectoryNet/TrajectoryNet/lib/layers").rglob("*.py"))),
)


@lru_cache(None)
def reference_module(name: str):
    paths = {"scbrain": REFERENCE_FILES[0], "spatial": REFERENCE_FILES[1],
             "scnode": REFERENCE_FILES[4], "scuniflow": REFERENCE_FILES[5]}
    if name == "scfit":
        root = str(ROOT / "reference_methods/scFiT/src")
        if root not in sys.path:
            sys.path.insert(0, root)
        from scfit import model
        return model
    spec = importlib.util.spec_from_file_location(f"nanobrain_upstream_{name}", ROOT / paths[name])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScBrainEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, dropout=0., **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
            reference_module("scbrain").FFN(n_layers=depth, model_dim=hidden_dim,
                                            expansion_factor=2.67, dropout=dropout))

    def forward(self, x):
        return self.net(x)


class ScNODEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, **_):
        super().__init__()
        self.net = reference_module("scnode").LinearNet(
            input_dim, [hidden_dim] * (depth - 1) or None, hidden_dim, "softplus")

    def forward(self, x):
        return self.net(x)


class ScviEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, dropout=0., **_):
        super().__init__()
        from scvi.nn import FCLayers
        self.net = FCLayers(n_in=input_dim, n_out=hidden_dim, n_layers=depth,
                           n_hidden=hidden_dim, dropout_rate=dropout,
                           use_batch_norm=False, use_layer_norm=True)

    def forward(self, x):
        return self.net(x)


class PanelPatchEncoder(nn.Module):
    """Signed standardized expression -> ordered gene-panel patches, no clipping."""
    def __init__(self, input_dim, hidden_dim, **_):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = min(16, input_dim)
        self.tokens = (input_dim + self.patch_size - 1) // self.patch_size
        self.projection = nn.Linear(self.patch_size, hidden_dim)
        self.gene_identity = nn.Parameter(torch.empty(1, self.tokens, hidden_dim))
        nn.init.normal_(self.gene_identity, std=.02)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, self.tokens * self.patch_size - self.input_dim))
        return self.projection(x.reshape(len(x), self.tokens, self.patch_size)) + self.gene_identity


class ScFiTAttention(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0., **_):
        super().__init__()
        self.block = reference_module("scfit").AdaLNTransformerBlock(
            hidden_dim, heads, 4 * hidden_dim, dropout=dropout, use_adaln=False)
        # Conditioning projections are unused in the unconditional encoder block.
        self.block.attn_mod = nn.Identity()
        self.block.ffn_mod = nn.Identity()

    def forward(self, x):
        squeeze = x.ndim == 2
        seq = x[:, None] if squeeze else x
        out, _ = self.block(seq, seq.new_zeros((len(seq), seq.shape[-1])), None)
        return out[:, 0] if squeeze else out


class BertAttention(nn.Module):
    """Geneformer's BERT block family, randomly initialized at nano width."""
    def __init__(self, hidden_dim, heads=4, dropout=0., **_):
        super().__init__()
        from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertLayer
        cfg = BertConfig(hidden_size=hidden_dim, num_attention_heads=heads,
                         intermediate_size=4 * hidden_dim, hidden_dropout_prob=dropout,
                         attention_probs_dropout_prob=dropout)
        cfg._attn_implementation = "sdpa"
        self.block = BertLayer(cfg)

    def forward(self, x):
        squeeze = x.ndim == 2
        out = self.block(x[:, None] if squeeze else x)[0]
        return out[:, 0] if squeeze else out


class TorchTransformer(nn.Module):
    """PyTorch TransformerEncoderLayer, also the standard scGPT block family."""
    def __init__(self, hidden_dim, heads=4, dropout=0., **_):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(hidden_dim, heads, 4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, x):
        squeeze = x.ndim == 2
        out = self.block(x[:, None] if squeeze else x)
        return out[:, 0] if squeeze else out


class ScBrainSpatial(nn.Module):
    def __init__(self, hidden_dim, **_):
        super().__init__()
        self.block = reference_module("spatial").GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, neighbors=None, neighbor_mask=None, **_):
        if neighbors is None or neighbor_mask is None:
            return x
        rows = torch.arange(len(x), device=x.device)[:, None].expand_as(neighbors)
        valid = neighbor_mask.bool() & (neighbors != rows)
        edges = torch.stack((neighbors[valid], rows[valid]))
        correction = self.block(x, edges)
        return x + torch.where(valid.any(1)[:, None], correction, torch.zeros_like(correction))


class ScFiTFlow(nn.Module):
    """Official scFiT AdaLN blocks over each cell's state/context token pair.

    Adapter supplies absolute start/end ages (years). Per-cell interpolation
    times stay distinct. This is a component hybrid, not full scFiT.
    """
    def __init__(self, hidden_dim, heads=4, depth=2, dropout=0., **_):
        super().__init__()
        upstream = reference_module("scfit")
        self.condition = upstream.SequenceConditionEncoder(hidden_dim, dropout=dropout)
        self.context = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(upstream.AdaLNTransformerBlock(
            hidden_dim, heads, 4 * hidden_dim, dropout=dropout) for _ in range(depth))
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, context, time, **_):
        if time.ndim == 1:
            time = time[:, None]
        if time.shape[1] == 1:
            time = torch.nn.functional.pad(time, (0, 2))
        if time.shape != (len(state), 3):
            raise ValueError("scfit_adaln flow requires [cells,3] time features")
        # Training samples per-cell tau, so use cells as B and two state/context
        # tokens. This respects every cell's own flow time without averaging it.
        condition = self.condition(time[:, :1], time[:, 1:2], time[:, 1:2] + time[:, 2:3])
        value = torch.stack((state, self.context(context)), dim=1)
        for block in self.blocks:
            value, _ = block(value, condition, None)
        return self.output(value[:, 0])


class TimeConditionedMLP(nn.Module):
    """scNODE LinearNet with explicit context and all three time coordinates."""
    def __init__(self, hidden_dim, depth=2, **_):
        super().__init__()
        self.net = reference_module("scnode").LinearNet(
            2 * hidden_dim + 3, [hidden_dim] * depth, hidden_dim, "tanh")

    def forward(self, state, context, time, **_):
        if time.ndim == 1:
            time = time[:, None]
        if time.shape[1] == 1:
            time = torch.nn.functional.pad(time, (0, 2))
        return self.net(torch.cat((state, context, time.to(state.dtype)), -1))


class ScBrainHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0., **_):
        super().__init__()
        self.net = reference_module("scbrain").Hierarchical_Decoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_num=[output_dim],
            dropout=dropout, apply_softmax=False)

    def forward(self, x):
        return self.net(x)[0]


class ScFiTHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0., **_):
        super().__init__()
        self.net = reference_module("scfit").TypeHead(hidden_dim, output_dim, dropout)

    def forward(self, x):
        return self.net(x)


class GraphSTSpatial(nn.Module):
    """Official GraphST encoding weight + PyTorch sparse A X W operation.

    The full contrastive/discriminator objective remains in the baseline;
    the composable primitive is its sparse graph embedding operator.
    """
    def __init__(self, hidden_dim, dropout=0., **_):
        super().__init__()
        from GraphST.model import Encoder_sparse
        encoder = Encoder_sparse(hidden_dim, hidden_dim, graph_neigh=None, dropout=dropout)
        self.weight = encoder.weight1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighbors=None, neighbor_mask=None, **_):
        if neighbors is None or neighbor_mask is None:
            return x
        rows = torch.arange(len(x), device=x.device)[:, None].expand_as(neighbors)
        valid = neighbor_mask.bool() & (neighbors != rows)
        index = torch.stack((rows[valid], neighbors[valid]))
        weights = valid.sum(1).clamp_min(1).float().reciprocal()[rows[valid]]
        # sparse.mm's mature CPU/CUDA implementation; keep sparse values and
        # dense operands FP32 for BF16 autocast portability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            adj = torch.sparse_coo_tensor(index, weights, (len(x), len(x))).coalesce()
            correction = torch.sparse.mm(adj, self.dropout(x).float() @ self.weight.float())
        return x + torch.where(valid.any(1)[:, None], correction.to(x.dtype), torch.zeros_like(x))


class TrajectoryNetFlow(nn.Module):
    def __init__(self, hidden_dim, depth=2, **_):
        super().__init__()
        root = str(ROOT / "reference_methods/TrajectoryNet")
        if root not in sys.path:
            sys.path.insert(0, root)
        from TrajectoryNet.lib.layers.odefunc import ODEnet
        self.context = nn.Linear(hidden_dim + 2, hidden_dim)
        self.net = ODEnet(hidden_dims=(hidden_dim,) * depth, input_shape=(hidden_dim,),
                          strides=None, conv=False, layer_type="concat", nonlinearity="tanh")

    def forward(self, state, context, time, **_):
        if time.ndim == 1:
            time = time[:, None]
        if time.shape[1] == 1:
            time = torch.nn.functional.pad(time, (0, 2))
        conditioned = state + self.context(torch.cat((context, time[:, 1:].to(state.dtype)), -1))
        return self.net(time[:, :1].to(state.dtype), conditioned)


class ScUniFlowField(nn.Module):
    """Official VectorFieldNet used as a directly trained latent velocity MLP.

    No spectral routing/Mamba/pretrained weight claim; learned dimension
    embeddings and source/age context align the official tensor interface.
    """
    def __init__(self, hidden_dim, **_):
        super().__init__()
        self.net = reference_module("scuniflow").VectorFieldNet(hidden_dim, hidden_dim)
        self.net.binary_mlp = nn.Identity()  # binary endpoint objective is not used
        self.dimension = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * .02)
        self.age = nn.Linear(2, hidden_dim)

    def forward(self, state, context, time, **_):
        if time.ndim == 1:
            time = time[:, None]
        if time.shape[1] == 1:
            time = torch.nn.functional.pad(time, (0, 2))
        condition = context + self.age(time[:, 1:].to(state.dtype))
        condition = condition[:, None, :] + self.dimension[None, :, :]
        return self.net(state, condition, time[:, :1].to(state.dtype))


def register_components():
    from .registry import registry
    entries = [
        ("encoder", "scbrain_swiglu", ScBrainEncoder, "scBrain FFN", "O(LBH²)", 2, 2),
        ("encoder", "scnode_mlp", ScNODEEncoder, "scNODE LinearNet", "O(LBH²)", 2, 2),
        ("encoder", "scvi_fc", ScviEncoder, "scvi.nn.FCLayers", "O(LBH²)", 2, 2),
        ("encoder", "panel_patches", PanelPatchEncoder, "torch.nn.Linear", "O(BGH)", 2, 3),
        ("attention", "scfit_block", ScFiTAttention, "scFiT AdaLNTransformerBlock (unconditional)", "O(BT²H)", None, None),
        ("attention", "bert", BertAttention, "transformers.BertLayer (random weights)", "O(BT²H)", None, None),
        ("attention", "torch_transformer", TorchTransformer, "torch.nn.TransformerEncoderLayer", "O(BT²H)", None, None),
        ("spatial", "scbrain_gcn", ScBrainSpatial, "scBrain GCNConv", "O(BKH+BH²)", 2, 2),
        ("spatial", "graphst_sparse", GraphSTSpatial, "GraphST Encoder_sparse.weight1 + torch.sparse.mm", "O(BKH+BH²)", 2, 2),
        ("flow", "scfit_adaln", ScFiTFlow, "scFiT SequenceConditionEncoder + AdaLNTransformerBlock", "O(LBH²)", 2, 2),
        ("flow", "scnode_time_mlp", TimeConditionedMLP, "scNODE LinearNet (conditional adapter)", "O(LBH²)", 2, 2),
        ("flow", "trajectorynet", TrajectoryNetFlow, "TrajectoryNet ODEnet (conditional CFM, no CNF density loss)", "O(LBH²)", 2, 2),
        ("flow", "scuniflow_field", ScUniFlowField, "scUniFlow VectorFieldNet (direct latent velocity adaptation)", "O(BH³)", 2, 2),
        ("head", "scbrain", ScBrainHead, "scBrain Hierarchical_Decoder (one level)", "O(BH²+BHC)", 2, 2),
        ("head", "scfit", ScFiTHead, "scFiT TypeHead", "O(BH²+BHC)", 2, 2),
    ]
    for kind, name, factory, source, complexity, irank, orank in entries:
        registry(kind).register(name, factory, source=source, complexity=complexity,
            input_rank=irank, output_rank=orank,
            notes="Trainable component hybrid, not a pretrained model or full baseline reproduction.")
