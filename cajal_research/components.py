"""Merge the existing nanoBrain library with Cajal's actual module interfaces.

The legacy source is frozen locally; no old dataset or trainer is imported. A
component adapter is not a reproduction of the source paper's complete method.
"""

import importlib
import inspect
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "component_library"
SLOTS = (
    "value_embedding",
    "token",
    "block_attention",
    "block_mixer",
    "latent",
    "classifier",
    "decoder",
    "neighborhood",
)
PARAM_RANGES = {
    "depth": ("integer", 1, 6),
    "dropout": ("number", 0.0, 0.5),
    "heads": ("integer", 1, 8),
    "expansion": ("number", 1.0, 8.0),
    "correction_init": ("number", 0.0, 1.0),
    "steps": ("integer", 1, 6),
    "bins": ("integer", 4, 64),
    "momentum": ("number", 0.0, 0.99),
    "eps": ("number", 1e-8, 1e-3),
}


@lru_cache(None)
def legacy():
    extras = ROOT / ".component_deps"
    if extras.is_dir() and str(extras) not in sys.path:
        sys.path.insert(0, str(extras))
    location = str(LIBRARY / "legacy/src")
    if location not in sys.path:
        sys.path.insert(0, location)
    importlib.import_module("nanobrain.modules")
    importlib.import_module("nanobrain.graph_ops")
    importlib.import_module("nanobrain.component_extensions")
    return importlib.import_module("nanobrain.registry")


@lru_cache(None)
def catalog():
    p = LIBRARY / "catalog.json"
    return json.loads(p.read_text()) if p.is_file() else []


def get(name):
    matches = [v for v in catalog() if v["id"] == name]
    if len(matches) != 1:
        raise ValueError("unknown merged component: " + name)
    return matches[0]


def validate_specs(specs):
    if not isinstance(specs, list):
        raise TypeError("components must be a list")
    occupied = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"slot", "name", "params"}:
            raise ValueError("component fields are slot/name/params")
        entry = get(spec["name"])
        if not entry.get("available") or spec["slot"] not in entry.get("slots", []):
            raise ValueError("component is unavailable or incompatible with this Cajal slot")
        if spec["slot"] != "token":
            if spec["slot"] in occupied:
                raise ValueError("only one component may replace a structural slot")
            occupied.add(spec["slot"])
        allowed = entry.get("parameter_ranges", {})
        if not isinstance(spec["params"], dict) or set(spec["params"]) - set(allowed):
            raise ValueError("unknown component parameter")
        for key, value in spec["params"].items():
            kind, low, high = allowed[key]
            if (kind == "integer" and type(value) is not int) or type(value) not in (int, float):
                raise ValueError("wrong component parameter type")
            if not low <= value <= high:
                raise ValueError("component parameter outside its declared range")


def specs_for(cfg, slot):
    return [v for v in cfg.research_components if v["slot"] == slot]


def _factory(entry, width, params, output_dim=None, input_dim=None):
    from brainwm_unified import model as M

    if entry["id"] == "cajal.swiglu":
        return M.SwiGLU(width, round(width * params.get("expansion", 1.0)))
    if entry["id"].startswith("legacy."):
        factory = legacy().registry(entry["legacy_kind"]).resolve(entry["legacy_name"]).factory
        kwargs = dict(params, hidden_dim=width, input_dim=width if input_dim is None else input_dim)
        if output_dim is not None:
            kwargs["output_dim"] = output_dim
        signature = inspect.signature(factory)
        if not any(v.kind is inspect.Parameter.VAR_KEYWORD for v in signature.parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return factory(**kwargs)
    raise ValueError("no tensor factory for " + entry["id"])


def token_module(spec, width):
    import torch

    entry = get(spec["name"])
    module = _factory(entry, width, spec["params"])

    class TokenAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = module

        def forward(self, x):
            if entry.get("flat_input"):
                return self.module(x.reshape(-1, x.shape[-1])).reshape_as(x)
            return self.module(x)

    return TokenAdapter()


def build_token_modules(cfg):
    import torch

    return torch.nn.ModuleList(token_module(s, cfg.d_model) for s in specs_for(cfg, "token"))


def install_value_embedding(encoder, cfg):
    import torch

    selected = specs_for(cfg, "value_embedding")
    if selected and selected[0]["name"] != "cajal.value_mlp":
        spec = selected[0]
        inner = _factory(get(spec["name"]), cfg.d_model, spec["params"], input_dim=1)

        class Values(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)[:, 0]

        encoder.value_emb = Values()


def install_block(block, cfg):
    import torch

    from brainwm_unified import model as M

    selected = specs_for(cfg, "block_attention")
    if selected and selected[0]["name"] != "cajal.attention":
        inner = token_module(selected[0], cfg.d_model)

        class IsolatedAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, x, mask):
                if isinstance(mask, M.JaggedLayout):
                    offsets = mask.offsets.tolist()
                elif isinstance(mask, torch.Tensor):
                    # Dense masks are only used by the small CPU numerical probe.
                    matrix = mask.reshape(-1, len(x), len(x))[0]
                    offsets, start = [0], 0
                    while start < len(x):
                        end = start + int(matrix[start].sum())
                        if end <= start:
                            break
                        offsets.append(end)
                        start = end
                else:
                    raise TypeError("alternate attention requires exact per-cell offsets")
                pieces = [self.inner(x[a:b].unsqueeze(0)).squeeze(0) for a, b in itertools.pairwise(offsets)]
                if offsets[-1] < len(x):
                    pieces.append(torch.zeros_like(x[offsets[-1] :]))
                return torch.cat(pieces)

        block.attn = IsolatedAttention()
    selected = specs_for(cfg, "block_mixer")
    if selected and selected[0]["name"] != "cajal.moe":
        spec = selected[0]
        inner = _factory(get(spec["name"]), cfg.d_model, spec["params"])

        class Mixer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, x, slot=0, gene=None):
                return self.inner(x)

        block.moe = Mixer()


def install_model(model, cfg):
    import torch

    selected = specs_for(cfg, "classifier")
    if selected and selected[0]["name"] != "cajal.hierarchical_head":
        spec = selected[0]
        entry = get(spec["name"])

        class Hierarchy(torch.nn.Module):
            def __init__(self, levels):
                super().__init__()
                self.heads = torch.nn.ModuleList(_classifier(entry, cfg.z_dim, n, spec["params"]) for n in levels)

            def observe(self, z, labels, maps):
                if self.training:
                    for head, mapping in zip(self.heads, maps):
                        if hasattr(head, "observe"):
                            head.observe(z, mapping[labels], None)

            def forward(self, z):
                return [head(z) for head in self.heads]

        model.cell_type_head, model.anatomy_head = Hierarchy(cfg.cell_type_levels), Hierarchy(cfg.anatomy_levels)
    selected = specs_for(cfg, "latent")
    if selected and selected[0]["name"] == "scvi.vae_encoder":
        from scvi.nn import Encoder

        params = selected[0]["params"]

        class Variational(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = Encoder(
                    cfg.d_model,
                    cfg.z_dim,
                    n_layers=params.get("depth", 2),
                    n_hidden=cfg.d_model,
                    dropout_rate=params.get("dropout", 0.1),
                    use_batch_norm=False,
                    use_layer_norm=True,
                    return_dist=True,
                )

            def forward(self, x):
                q, _ = self.net(x)
                return torch.cat((q.loc, q.scale.square().clamp_min(1e-12).log()), -1)

        model.to_latent = Variational()
    selected = specs_for(cfg, "decoder")
    if selected and selected[0]["name"] == "scvi.nb_decoder":
        from scvi.nn import DecoderSCVI

        params = selected[0]["params"]

        class NBDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = DecoderSCVI(
                    cfg.z_dim,
                    cfg.n_genes,
                    n_cat_list=[cfg.n_batches],
                    n_layers=params.get("depth", 2),
                    n_hidden=cfg.decoder_hidden,
                    use_batch_norm=False,
                    use_layer_norm=True,
                )
                self.net.px_dropout_decoder.requires_grad_(False)
                self.log_theta = torch.nn.Parameter(torch.zeros(cfg.n_genes))

            def forward(self, z, batch, task="reconstruct"):
                scale, _, _, _ = self.net("gene", z, z.new_zeros((len(z), 1)), batch[:, None])
                return scale.clamp_min(1e-12).log()

        model.decoder = NBDecoder()
    selected = specs_for(cfg, "neighborhood")
    if selected and selected[0]["name"] != "cajal.neighborhood_transformer":
        spec = selected[0]
        entry = get(spec["name"])

        class Neighborhood(torch.nn.Module):
            def __init__(self):
                super().__init__()
                d = cfg.spatial_d
                self.inp = torch.nn.Linear(cfg.z_dim, d)
                self.pos = torch.nn.Sequential(torch.nn.Linear(3, d), torch.nn.SiLU(), torch.nn.Linear(d, d))
                self.mask_emb = torch.nn.Parameter(torch.zeros(d))
                self.graph = _factory(entry, d, spec["params"])
                self.out = torch.nn.Linear(d, cfg.z_dim)

            def forward(self, z, rel, masked, pad):
                h = torch.where(masked.unsqueeze(-1), self.mask_emb, self.inp(z)) + self.pos(rel)
                outputs = []
                for group, coordinates, missing in zip(h, rel, pad):
                    ids = (~missing).nonzero().flatten()
                    values = group[ids]
                    # Geometry stays inside the one Cajal neighbourhood, never across groups.
                    distance = torch.cdist(coordinates[ids].float(), coordinates[ids].float())
                    distance.fill_diagonal_(float("inf"))
                    count = max(1, len(ids) - 1)
                    neighbors = distance.argsort(-1)[:, :count]
                    valid = torch.isfinite(distance.gather(1, neighbors))
                    value = self.graph(values, neighbors=neighbors, neighbor_mask=valid)
                    output = group.new_zeros(group.shape).index_copy(0, ids, value)
                    outputs.append(output)
                return self.out(torch.stack(outputs))

        model.neighborhood = Neighborhood()


def _classifier(entry, width, output_dim, params):
    if entry["id"] == "scanvi.classifier":
        from scvi.module._classifier import Classifier

        return Classifier(
            width,
            n_labels=output_dim,
            n_layers=params.get("depth", 2),
            n_hidden=max(32, width),
            dropout_rate=params.get("dropout", 0.1),
            logits=True,
            use_batch_norm=False,
            use_layer_norm=True,
        )
    return _factory(entry, width, params, output_dim)


def observe_heads(model, batch, out, maps):
    for task in ("cell_type", "anatomy"):
        head = getattr(model, task + "_head")
        if hasattr(head, "observe"):
            head.observe(out["z"], batch[task], maps[task])
