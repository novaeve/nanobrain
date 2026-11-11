"""Trainable adapters for the 17 collected upstream model components.

Only shape alignment/conditioning is added here. Source math is loaded from
pinned upstream files; these are component hybrids, not whole-method replicas.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from .upstream_loader import selected, whole_file

TOKEN_ENCODERS = {"scfoundation_autobin", "scprint_value"}


class ScTabEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps=3, depth=2, dropout=0.0):
        super().__init__()
        if not 1 <= steps <= 8:
            raise ValueError("TabNet steps must be in [1,8]")
        self.net = whole_file("scTab", "cellnet/tabnet/tab_network.py").TabNetEncoder(
            input_dim, hidden_dim, n_d=hidden_dim, n_a=hidden_dim, n_steps=steps,
            n_independent=depth, n_shared=depth, virtual_batch_size=128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        steps, _ = self.net(x)
        return self.dropout(torch.stack(steps).sum(0))


class ScTabGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = whole_file("scTab", "cellnet/tabnet/tab_network.py").GLU_Layer(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.net(x)


class ScNymEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, dropout=0.0):
        super().__init__()
        original = selected("scNym", "scnym/model.py", ("ResBlock", "CellTypeCLF"))
        full = original["CellTypeCLF"](n_genes=input_dim, n_cell_types=2,
            n_hidden=hidden_dim, n_hidden_init=hidden_dim, n_layers=depth, init_dropout=0.)
        # The original classifier exposes its feature subnetwork separately.
        self.net = full.embed
        for layer in self.net.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dropout

    def forward(self, x):
        return self.net(x)


class ScNymResidual(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = selected("scNym", "scnym/model.py", ("ResBlock",))["ResBlock"](hidden_dim, hidden_dim)

    def forward(self, x):
        return self.net(x)


def edges_from_neighbors(x, neighbors, neighbor_mask):
    if neighbors is None or neighbor_mask is None:
        return None, None
    target = torch.arange(len(x), device=x.device)[:, None].expand_as(neighbors)
    valid = neighbor_mask.bool() & (neighbors >= 0) & (neighbors < len(x)) & (neighbors != target)
    edge = torch.stack((neighbors[valid], target[valid]))
    return edge, valid.any(1)


class STAGATESpatial(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = whole_file("STAGATE", "STAGATE_pyG/gat_conv.py").GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, add_self_loops=True, bias=False)

    def forward(self, x, neighbors=None, neighbor_mask=None, **_):
        edge, has = edges_from_neighbors(x, neighbors, neighbor_mask)
        if edge is None or not bool(has.any()):
            return x
        out = self.net(x, edge)
        return torch.where(has[:, None], out, x)


class SEDREncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = selected("SEDR", "SEDR/SEDR_module.py", ("full_block",))["full_block"](
            input_dim, hidden_dim, dropout)

    def forward(self, x):
        return self.net(x)


class SEDRSpatial(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = selected("SEDR", "SEDR/SEDR_module.py", ("GraphConvolution",))["GraphConvolution"](
            hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, x, neighbors=None, neighbor_mask=None, **_):
        edge, has = edges_from_neighbors(x, neighbors, neighbor_mask)
        if edge is None or not bool(has.any()):
            return x
        self_edge = torch.arange(len(x), device=x.device)
        index = torch.cat((edge.flip(0), torch.stack((self_edge, self_edge))), dim=1)
        degree = torch.bincount(index[0], minlength=len(x)).float().clamp_min(1)
        weights = degree[index[0]].rsqrt() * degree[index[1]].rsqrt()
        adj = torch.sparse_coo_tensor(index, weights.to(x), (len(x), len(x))).coalesce()
        out = self.net(x, adj)
        return torch.where(has[:, None], out, x)


class SpaGCNSpatial(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = whole_file("SpaGCN", "SpaGCN_package/SpaGCN/layers.py").GraphConvolution(hidden_dim, hidden_dim)

    def forward(self, x, neighbors=None, neighbor_mask=None, **_):
        edge, has = edges_from_neighbors(x, neighbors, neighbor_mask)
        if edge is None or not bool(has.any()):
            return x
        # Reuse verified one-hop sparse adjacency; do not invent unavailable
        # histology or context coordinates. No extra spatial hop is added.
        ids = torch.arange(len(x), device=x.device)
        index = torch.cat((edge.flip(0), torch.stack((ids, ids))), dim=1)
        degree = torch.bincount(index[0], minlength=len(x)).float().clamp_min(1)
        weights = degree[index[0]].reciprocal().to(x)
        adj = torch.sparse_coo_tensor(index, weights, (len(x), len(x))).coalesce()
        return torch.where(has[:, None], self.net(x, adj), x)


class ScPoliEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_count=64, condition_dim=8, dropout=0.0):
        super().__init__()
        self.condition_count = condition_count
        self.embedding = nn.Embedding(condition_count + 1, condition_dim)
        original = selected("scPoli", "scarches/models/scpoli/scpoli.py", ("Encoder", "CondLayers"))
        with redirect_stdout(io.StringIO()):
            self.net = original["Encoder"]([input_dim, hidden_dim], hidden_dim,
                use_bn=False, use_ln=True, use_dr=dropout > 0, dr_rate=dropout, embedding_dim=condition_dim)
        # Mean-only deterministic adapter; unused variance parameters are frozen.
        for p in self.net.log_var_encoder.parameters():
            p.requires_grad_(False)
            p._nanobrain_always_frozen = True

    def forward(self, x, batch_id):
        ids = batch_id.reshape(-1).long()
        if bool((ids >= self.condition_count).any()):
            raise ValueError("condition_count does not cover technical batch IDs")
        ids = ids.masked_fill(ids < 0, self.condition_count)
        mean, _ = self.net(x, self.embedding(ids))
        return mean


class ScPoliPrototype(nn.Module):
    def __init__(self, hidden_dim, output_dim, momentum=0.9):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("prototype momentum must be in [0,1)")
        self.momentum = momentum
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.register_buffer("prototypes", torch.zeros(output_dim, hidden_dim))
        self.register_buffer("counts", torch.zeros(output_dim, dtype=torch.long))
        self._classify = selected("scPoli", "scarches/models/scpoli/scpoli.py", ("scpoli",))["scpoli"].classify

    @torch.no_grad()
    def observe(self, embedding, labels, mask):
        if not self.training:
            raise RuntimeError("prototype updates require training mode")
        valid = labels.ge(0) & labels.lt(len(self.counts))
        if mask is not None:
            valid &= mask.bool()
        for label in labels[valid].unique():
            values = embedding[valid & labels.eq(label)].detach().float()
            average = values.mean(0).to(self.prototypes)
            if self.counts[label] == 0:
                self.prototypes[label].copy_(average)
            else:
                self.prototypes[label].lerp_(average, 1 - self.momentum)
            self.counts[label] += len(values)

    def forward(self, x):
        if not bool(self.counts.any()):
            raise ValueError("prototype head needs train-only observation before prediction")
        with torch.autocast(device_type=x.device.type, enabled=False):
            self.prototypes_labeled = {"mean": self.prototypes.float()}
            _, _, distances = self._classify(self, x.float(), prototype=True,
                classes_list=torch.arange(len(self.counts), device=x.device), get_prob=False, log_distance=False)
        logits = -distances * self.log_temperature.clamp(-4, 4).exp()
        return logits.masked_fill(self.counts[None, :] == 0, -10000.)


class MIOFlowVelocity(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.condition = nn.Linear(hidden_dim + 3, hidden_dim)
        self.net = whole_file("MIOFlow", "mioflow/core/models/ode_model.py").ODEFunc(
            hidden_dim, hidden_dim, momentum_beta=0.)

    def forward(self, state, context, time):
        conditioned = state + self.condition(torch.cat((context, time.to(context)), -1))
        return self.net(time[:, :1].to(state), conditioned)


class PRESCIENTVelocity(nn.Module):
    def __init__(self, hidden_dim, depth=2):
        super().__init__()
        self.condition = nn.Linear(hidden_dim + 3, hidden_dim)
        cls = selected("PRESCIENT", "prescient/train/model.py", ("AutoGenerator",))["AutoGenerator"]
        self.net = cls(SimpleNamespace(x_dim=hidden_dim, k_dim=hidden_dim, layers=depth, activation="softplus"))

    def forward(self, state, context, time):
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            conditioned = state.float() + self.condition(torch.cat((context.float(), time.float()), -1))
            if not conditioned.requires_grad:
                conditioned = conditioned.requires_grad_()
            velocity = self.net._drift(conditioned)
        return velocity.to(state) if outer_grad else velocity.detach().to(state)


class CellOTEndpoint(nn.Module):
    def __init__(self, hidden_dim, depth=2):
        super().__init__()
        self.condition = nn.Linear(hidden_dim + 3, hidden_dim)
        cls = whole_file("CellOT", "cellot/networks/icnns.py").ICNN
        self.net = cls(input_dim=hidden_dim, hidden_units=[hidden_dim] * depth, softplus_W_kernels=True)

    def forward(self, state, context, time):
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            conditioned = state.float() + self.condition(torch.cat((context.float(), time.float()), -1))
            if not conditioned.requires_grad:
                conditioned = conditioned.requires_grad_()
            endpoint = self.net.transport(conditioned)
        return endpoint.to(state) if outer_grad else endpoint.detach().to(state)


class AutoBinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bins=16):
        super().__init__()
        cls = selected("scFoundation", "model/pretrainmodels/mae_autobin.py", ("AutoDiscretizationEmbedding2",))["AutoDiscretizationEmbedding2"]
        self.net = cls(hidden_dim, input_dim, bins, 1., mask_token_id=-1.e30, pad_token_id=-2.e30)
        self.gene_identity = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * .02)

    def forward(self, x):
        return self.net(x.unsqueeze(-1)) + self.gene_identity


class ScPRINTValue(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0, depth=1):
        super().__init__()
        cls = selected("scPRINT", "scprint/model/encoders.py", ("ContinuousValueEncoder",))["ContinuousValueEncoder"]
        self.net = cls(hidden_dim, dropout=dropout, layers=depth)
        self.gene_identity = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * .02)

    def forward(self, x):
        if bool((x < 0).any()):
            raise ValueError("scPRINT value encoder needs declared nonnegative expression input")
        return self.net(x) + self.gene_identity


class ScPRINTHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, depth=2, dropout=0.0):
        super().__init__()
        cls = selected("scPRINT", "scprint/model/decoders.py", ("ClsDecoder",))["ClsDecoder"]
        self.net = cls(hidden_dim, output_dim, layers=[hidden_dim] * depth, dropout=dropout)

    def forward(self, x):
        return self.net(x)


class ScBERTAttention(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.0):
        super().__init__()
        self.net = whole_file("scBERT", "performer_pytorch/performer_pytorch.py").SelfAttention(
            dim=hidden_dim, heads=heads, dim_head=hidden_dim // heads, dropout=dropout)

    def forward(self, x):
        squeezed = x.ndim == 2
        out = self.net(x[:, None] if squeezed else x)
        return out[:, 0] if squeezed else out


def register_extensions():
    from .registry import registry
    entries = [
        ("encoder", "sctab_tabnet", ScTabEncoder, "scTab TabNetEncoder"),
        ("operation", "sctab_glu", ScTabGLU, "scTab GLU_Layer"),
        ("spatial", "stagate_gat", STAGATESpatial, "STAGATE custom GATConv, one hop, heads=1"),
        ("encoder", "scpoli_conditional", ScPoliEncoder, "scPoli Encoder + technical condition embedding; deterministic mean"),
        ("head", "scpoli_prototype", ScPoliPrototype, "scPoli classify; train-only EMA prototypes, checkpointed buffers"),
        ("flow", "mioflow_ode", MIOFlowVelocity, "MIOFlow ODEFunc; stateless conditional adapter"),
        ("flow", "prescient_potential", PRESCIENTVelocity, "PRESCIENT AutoGenerator._drift; original positive gradient"),
        ("encoder", "scnym_mlp", ScNymEncoder, "scNym CellTypeCLF.embed"),
        ("residual", "scnym_resblock", ScNymResidual, "scNym ResBlock"),
        ("encoder", "sedr_mlp", SEDREncoder, "SEDR full_block"),
        ("spatial", "sedr_gcn", SEDRSpatial, "SEDR GraphConvolution; one-hop normalized sparse graph"),
        ("spatial", "spagcn_gcn", SpaGCNSpatial, "SpaGCN GraphConvolution; existing verified one-hop adjacency, no histology"),
        ("encoder", "scfoundation_autobin", AutoBinEncoder, "scFoundation AutoDiscretizationEmbedding2; new gene identity"),
        ("encoder", "scprint_value", ScPRINTValue, "scPRINT ContinuousValueEncoder; nonnegative expression only"),
        ("head", "scprint_classifier", ScPRINTHead, "scPRINT ClsDecoder"),
        ("attention", "scbert_performer", ScBERTAttention, "scBERT SelfAttention / FastAttention"),
        ("endpoint", "cellot_icnn_transport", CellOTEndpoint, "CellOT ICNN.transport; convex softplus weight parameterization"),
    ]
    for kind, name, factory, source in entries:
        registry(kind).register(name, factory, source=source,
            notes="Pinned upstream structure adapter; no pretrained weights or original training recipe.",
            input_rank=2 if kind in {"encoder", "spatial", "head", "residual", "endpoint"} else None,
            output_rank=3 if name in TOKEN_ENCODERS else (None if kind == "attention" else 2))


register_extensions()
