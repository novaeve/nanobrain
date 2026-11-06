"""Tensor-only extensions; the same constructor is used by training and inference."""

import torch

from .code_policy import load_tensor_code


def entries(payload, kind):
    return [p for p in payload.get("plugins", []) if p["kind"] == kind]


def factory(payload, p):
    entry = "compute" if p["kind"].endswith("loss") else "build"
    return load_tensor_code(payload["files"][p["file"]], entry, p["name"])


def settings(p):
    return {k: p["settings"].get(k, spec["default"]) for k, spec in p["parameters"].items()}


def build_transforms(payload, hidden_dim):
    modules = torch.nn.ModuleList()
    for p in entries(payload, "token_transform"):
        module = factory(payload, p)(hidden_dim=hidden_dim, **settings(p))
        if not isinstance(module, torch.nn.Module):
            raise TypeError("token transform must return nn.Module")
        modules.append(module)
    return modules


def apply_transforms(modules, h, doc):
    # A plugin never sees two cells together, labels, masks, neighbours or datasets.
    # Padding is left untouched. Even a custom attention cannot mix different cells.
    if not len(modules):
        return h
    chunks = []
    for indices in torch.split(
        torch.arange(len(doc), device=doc.device), torch.unique_consecutive(doc, return_counts=True)[1].tolist()
    ):
        part = h[indices]
        if int(doc[indices[0]]) >= 0:
            original = part.shape
            for module in modules:
                part = module(part.unsqueeze(0)).squeeze(0)
                if part.shape != original or not torch.isfinite(part).all():
                    raise ValueError("token transform changed shape or produced nonfinite values")
        chunks.append(part)
    return torch.cat(chunks)


def loss_for(payload, kind, prediction, target):
    selected = entries(payload, kind)
    if not selected:
        return None
    p = selected[0]
    value = factory(payload, p)(prediction, target, **settings(p))
    if not isinstance(value, torch.Tensor) or value.ndim or not torch.isfinite(value) or value < 0:
        raise ValueError("custom loss must be a finite nonnegative scalar")
    if torch.is_grad_enabled() and prediction.requires_grad and not value.requires_grad:
        raise ValueError("custom loss must retain the prediction gradient")
    return value


def replace_losses(payload, model, b, out, level_maps, losses):
    if not payload:
        return losses
    gene = loss_for(payload, "gene_loss", out["gene_pred"], b["tok_target"][b["tok_masked"]].float())
    if gene is not None:
        losses["gene"] = gene
    for task in ("cell_type", "anatomy"):
        if entries(payload, task + "_loss"):
            logits = getattr(model, task + "_head")(out["z"])
            losses[task] = torch.stack(
                [
                    loss_for(payload, task + "_loss", lg.float(), mapping[b[task]])
                    for lg, mapping in zip(logits, level_maps[task])
                ]
            ).mean()
    return losses


def custom_optimizer(payload, parameters, lr, weight_decay):
    (p,) = entries(payload, "optimizer")
    parameters = list(parameters)
    opt = factory(payload, p)(parameters=parameters, lr=lr, weight_decay=weight_decay, **settings(p))
    if not isinstance(opt, torch.optim.Optimizer):
        raise TypeError("custom optimizer must return torch.optim.Optimizer")
    actual = [id(v) for group in opt.param_groups for v in group["params"]]
    expected = [id(v) for v in parameters]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("optimizer must cover every trainable parameter exactly once")
    return opt


def make_optimizers(model, cfg, muon, groups):
    if cfg.optimizer in muon:
        a, b = groups
        return [
            muon[cfg.optimizer](a, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(b, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True),
        ]
    parameters = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "custom":
        return [custom_optimizer(cfg.research_plugins, parameters, cfg.lr, cfg.weight_decay)]
    cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[cfg.optimizer]
    return [cls(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)]
