"""Unit tests on the GPU, on a synthetic subset written in the data contract format."""

import json

import numpy as np
import pytest
import torch

from brainwm_unified import model as M
from brainwm_unified.config import DataConfig, ModelConfig
from brainwm_unified.data import Subset, build_batch, load_level_maps
from brainwm_unified.objectives import TASKS, compute_losses

N_GENES, N_TYPES, N_REGIONS, K = 50, 6, 4, 4
DEVICE = torch.device("cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def subset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("subset")
    rng = np.random.default_rng(0)
    n_diss, n_sp = 40, 24
    n = n_diss + n_sp
    panels = [np.arange(N_GENES), np.arange(20)]
    panel_id = np.r_[np.zeros(n_diss, int), np.ones(n_sp, int)]
    cell_type = rng.integers(0, N_TYPES, n)
    values, genes, ptr = [], [], [0]
    for c in range(n):
        g = panels[panel_id[c]]
        lam = 1.0 + 4.0 * (g % N_TYPES == cell_type[c])
        x = rng.poisson(lam).astype(np.float32)
        keep = x > 0
        values.append(x[keep])
        genes.append(g[keep])
        ptr.append(ptr[-1] + int(keep.sum()))
    spatial = np.full((n, 3), np.nan, np.float32)
    spatial[n_diss:] = rng.uniform(0, 100, (n_sp, 3)).astype(np.float32)
    spatial[n_diss:, 2] = 0
    neighbors = np.full((n, K), -1, np.int64)
    neighbor_dist = np.full((n, K), np.nan, np.float32)
    sp = np.arange(n_diss, n)
    for c in sp:
        d = np.linalg.norm(spatial[sp] - spatial[c], axis=1)
        order = [i for i in np.argsort(d) if sp[i] != c][:K]
        neighbors[c] = sp[order]
        neighbor_dist[c] = d[order]
    arrays = {
        "values": np.concatenate(values),
        "gene_index": np.concatenate(genes).astype(np.int32),
        "row_ptr": np.asarray(ptr, np.int64),
        "cell_type": cell_type,
        "anatomy": rng.integers(0, N_REGIONS, n),
        "integration_batch": rng.integers(0, 3, n),
        "panel_id": panel_id,
        "spatial": spatial,
        "neighbors": neighbors,
        "neighbor_dist": neighbor_dist,
        "panel_gene_ptr": np.array([0, N_GENES, N_GENES + 20]),
        "panel_gene_index": np.concatenate(panels),
    }
    for k, v in arrays.items():
        np.save(root / f"{k}.npy", v)
    cell_types = [{"index": i, "parent_lineage_index": i % 2} for i in range(N_TYPES)]
    (root / "cell_type_vocab.json").write_text(
        json.dumps({"levels": ["lineage", "cell_type"], "lineage": ["a", "b"], "cell_type": cell_types})
    )
    regions = [{"index": i} for i in range(N_REGIONS)]
    (root / "anatomy_vocab.json").write_text(json.dumps({"levels": ["anatomy"], "anatomy": regions}))
    return str(root)


def tiny_config(**kw):
    cfg = ModelConfig(
        n_genes=N_GENES,
        n_batches=3,
        cell_type_levels=[2, N_TYPES],
        anatomy_levels=[N_REGIONS],
        d_model=32,
        n_heads=4,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        loops=2,
        min_loops=1,
        max_loops=3,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        compile_blocks=False,
        z_dim=8,
        decoder_hidden=32,
        spatial_d=16,
        spatial_layers=1,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def make(subset_dir, seed=0, **kw):
    ds = Subset(subset_dir)
    dcfg = DataConfig(root=subset_dir, dissociated_cells=8, spatial_groups=3, neighbors=K, zero_tokens=4)
    b = {k: v.to(DEVICE) for k, v in build_batch(ds, dcfg, N_GENES, np.random.default_rng(seed)).items()}
    maps = {
        name: [torch.as_tensor(m, device=DEVICE) for m in load_level_maps(f"{subset_dir}/{name}_vocab.json")]
        for name in ("cell_type", "anatomy")
    }
    return ds, b, maps, ds.panel_matrix(N_GENES).to(DEVICE)


def test_batch_invariants(subset_dir):
    _, b, _, panels = make(subset_dir)
    n_cells = int(b["tok_cls"].sum())
    assert b["cell_type"].shape[0] == n_cells == len(b["cnt_ptr"]) - 1
    assert not (b["tok_masked"] & b["tok_cls"]).any()
    assert (b["tok_in"][b["tok_masked"]] == 0).all()
    # every non-CLS token is a gene measured by that cell's panel
    doc = b["tok_doc"][~b["tok_cls"]]
    assert panels[b["panel"][doc], b["tok_gene"][~b["tok_cls"]]].all()
    m = b["grp_members"]
    assert (m[:, 0] >= 0).all() and (m < n_cells).all()
    assert b["grp_masked"][m < 0].sum() == 0 and (b["grp_masked"].sum(1) >= 1).all()


@pytest.mark.parametrize("nbeats,decoder", [(True, "separate"), (False, "separate"), (True, "unified")])
def test_forward_backward_finite(subset_dir, nbeats, decoder):
    _, b, maps, panels = make(subset_dir)
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(nbeats=nbeats, decoder=decoder)).to(DEVICE)
    out = model(b, M.dense_mask(b["tok_doc"]), loops=3)
    losses, _ = compute_losses(model, b, out, panels, maps, kl_weight=1.0)
    assert set(losses) == set(TASKS)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_unified_decoder_conditions_every_expression_output_on_batch(subset_dir):
    """Imputed values and decoded logits change with the batch; the separate design keeps its parameters."""
    _, b, _, _ = make(subset_dir)
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(decoder="unified")).to(DEVICE).eval()
    torch.nn.init.normal_(model.decoder.film.weight, std=0.5)  # FiLM starts at zero; give it something to do
    with torch.no_grad():
        out = model(b, M.dense_mask(b["tok_doc"]))
        other = dict(b, batch=(b["batch"] + 1) % 3)
        moved = model(other, M.dense_mask(b["tok_doc"]))
        assert out["gene_pred"].shape == (int(b["tok_masked"].sum()),)
        assert torch.equal(out["mu"], moved["mu"])  # the encoder never sees the batch
        assert not torch.allclose(out["gene_pred"], moved["gene_pred"])
        z = out["mu"]
        assert model.decoder(z, b["batch"]).shape == (len(z), model.cfg.n_genes)
        assert not torch.allclose(model.decoder(z, b["batch"]), model.decoder(z, b["batch"], task="spatial"))
    separate = M.UnifiedModel(tiny_config()).state_dict()
    assert any(k.startswith("gene_head.") for k in separate) and "decoder.batch_emb.weight" in separate
    with pytest.raises(ValueError, match="decoder"):
        M.UnifiedModel(tiny_config(decoder="both"))


def test_spatial_latent_target_trains_the_predictor_not_the_target(subset_dir):
    """With spatial_latent the spatial loss gains a stop-gradient latent term; off, it is unchanged."""
    _, b, maps, panels = make(subset_dir)
    losses = {}
    for weight in (0.0, 1.0):
        torch.manual_seed(0)
        model = M.UnifiedModel(tiny_config(spatial_latent=weight)).to(DEVICE).eval()
        out = model(b, M.dense_mask(b["tok_doc"]))
        losses[weight], metrics = compute_losses(model, b, out, panels, maps, kl_weight=1.0)
        assert ("spatial/latent_mse" in metrics) == bool(weight)
    extra = losses[1.0]["spatial"] - losses[0.0]["spatial"]
    assert torch.isclose(extra, metrics["spatial/latent_mse"]) and extra > 0
    for k in TASKS:  # the other tasks do not move
        if k != "spatial":
            torch.testing.assert_close(losses[1.0][k], losses[0.0][k])
    # a hidden cell that is visible in no group reaches the latent term only as a target: no gradient
    members, masked = b["grp_members"], b["grp_masked"]
    target = members[masked]
    only_target = target[~torch.isin(target, members[~masked & (members >= 0)])]
    (grad,) = torch.autograd.grad(metrics["spatial/latent_mse"], out["mu"], retain_graph=True)
    assert grad[only_target].abs().sum() == 0


def test_padding_does_not_change_outputs(subset_dir):
    _, b, _, _ = make(subset_dir)
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config()).to(DEVICE).eval()
    padded = M.pad_tokens(b, 128)
    assert padded["tok_gene"].shape[0] % 128 == 0
    a = model(b, M.dense_mask(b["tok_doc"]))["mu"]
    p = model(padded, M.dense_mask(padded["tok_doc"]))["mu"]
    torch.testing.assert_close(a, p, rtol=1e-4, atol=1e-5)


def test_all_five_losses_decrease(subset_dir):
    """Overfitting one fixed batch: every objective must go down under joint training."""
    _, b, maps, panels = make(subset_dir, seed=1)
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    mask = M.dense_mask(b["tok_doc"])
    history = []
    for _ in range(120):
        out = model(b, mask, loops=2)
        losses, _ = compute_losses(model, b, out, panels, maps, kl_weight=0.1)
        history.append({t: losses[t].item() for t in TASKS})
        opt.zero_grad()
        sum(losses[t] / history[0][t] for t in TASKS).backward()
        opt.step()
        for moe in model.encoder.moe_layers():
            moe.update_bias(1e-3)
    for t in TASKS:
        first, last = history[0][t], np.mean([h[t] for h in history[-10:]])
        assert last < 0.9 * first, (t, first, last)


def test_router_bias_moves_towards_balance():
    moe = M.MoE(8, tiny_config()).to(DEVICE)
    moe.load[0] = torch.tensor([10.0, 0.0, 0.0, 2.0], device=DEVICE)
    moe.update_bias(0.1)
    assert moe.route_bias[0, 0] < 0 and moe.route_bias[0, 1] > 0
    assert moe.load.sum() == 0


def test_param_groups_cover_all_parameters():
    model = M.UnifiedModel(tiny_config()).to(DEVICE)
    muon, adamw = M.param_groups(model)
    assert all(p.ndim in (2, 3) for p in muon) and any(p.ndim == 3 for p in muon)  # stacked experts
    assert len(muon) + len(adamw) == len(list(model.parameters()))


def test_hash_routing_depends_only_on_gene():
    cfg = tiny_config()
    moe = M.MoE(8, cfg, hash_route=True).to(DEVICE)
    gene = torch.tensor([3, 3, 7], device=DEVICE)
    assert torch.equal(moe.hash_table[3], moe.hash_table[gene[1]])
    assert len(set(moe.hash_table[7].tolist())) == cfg.top_k  # distinct experts per gene
    torch.manual_seed(0)
    a, b = moe(torch.randn(3, 8, device=DEVICE), gene=gene), moe(torch.randn(3, 8, device=DEVICE), gene=gene)
    assert a.shape == b.shape == (3, 8)
    assert torch.equal(moe.load[0], torch.bincount(moe.hash_table[gene].flatten(), minlength=cfg.n_experts) * 2.0)


def test_probe_groups_cover_encoder_layers():
    from brainwm_unified.train import probe_group

    model = M.UnifiedModel(tiny_config()).to(DEVICE)
    groups = {probe_group(n) for n, _ in model.encoder.named_parameters()}
    assert {"embed", "prelude.0", "core.0", "coda.0"} <= groups
    assert probe_group("core.1.moe.experts.3.w1.weight") == "core.1"


def test_grouped_muon_matches_torch_muon():
    from brainwm_unified.optim import GroupedMuon

    torch.manual_seed(0)
    shapes = [(64, 32), (64, 32), (64, 32), (32, 64), (32, 64), (48, 48)]
    ref = [torch.nn.Parameter(torch.randn(s, device=DEVICE)) for s in shapes]
    ours = [torch.nn.Parameter(p.detach().clone()) for p in ref]
    kw = {"lr": 0.02, "weight_decay": 0.01, "adjust_lr_fn": "match_rms_adamw"}
    opt_ref, opt_ours = torch.optim.Muon(ref, **kw), GroupedMuon(ours, **kw)
    for _ in range(5):
        grads = [torch.randn(s, device=DEVICE) for s in shapes]
        for p, q, g in zip(ref, ours, grads):
            p.grad, q.grad = g.clone(), g.clone()
        opt_ref.step()
        opt_ours.step()
    for p, q in zip(ref, ours):
        torch.testing.assert_close(q, p, rtol=0, atol=2e-3)


def test_moe_matches_per_token_reference():
    """Fused grouped experts + gather-only dispatch/combine equal an explicit per-token loop, values and grads."""
    torch.manual_seed(0)
    cfg = tiny_config(moe_backend="loop")  # fp32 experts: checks dispatch/combine exactly; backends agree in test_gpu
    moe = M.MoE(16, cfg).to(DEVICE)
    x = torch.randn(9, 16, device=DEVICE, requires_grad=True)
    out = moe(x)
    out.square().sum().backward()
    got = (out.detach(), x.grad.clone(), moe.w13.grad.clone(), moe.w2.grad.clone())
    x.grad, moe.w13.grad, moe.w2.grad = None, None, None

    scores = torch.sqrt(torch.nn.functional.softplus(moe.routers[0](x)))
    idx = torch.topk(scores + moe.route_bias[0], cfg.top_k, dim=-1).indices
    gate = scores.gather(-1, idx)
    gate = gate / gate.sum(-1, keepdim=True)
    h = cfg.expert_hidden
    rows = []
    for t in range(x.shape[0]):
        row = moe.shared(x[t : t + 1])[0]
        for j in range(cfg.top_k):
            e = idx[t, j]
            a = moe.w13[e] @ x[t]
            row = row + gate[t, j] * (moe.w2[e] @ (torch.nn.functional.silu(a[:h]) * a[h:]))
        rows.append(row)
    ref = torch.stack(rows)
    ref.square().sum().backward()
    for a, b in zip(got, (ref.detach(), x.grad, moe.w13.grad, moe.w2.grad)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_grouped_muon_splits_stacked_experts_like_separate_matrices():
    from brainwm_unified.optim import GroupedMuon

    torch.manual_seed(0)
    e, h, d = 3, 8, 16
    w13 = torch.nn.Parameter(torch.randn(e, 2 * h, d, device=DEVICE))
    w13.muon_split = 2
    separate = [torch.nn.Parameter(w13.detach()[i, s * h : (s + 1) * h].clone()) for i in range(e) for s in range(2)]
    kw = {"lr": 0.02, "weight_decay": 0.01, "adjust_lr_fn": "match_rms_adamw"}
    opt_ours, opt_ref = GroupedMuon([w13], **kw), torch.optim.Muon(separate, **kw)
    for _ in range(3):
        g = torch.randn(e, 2 * h, d, device=DEVICE)
        w13.grad = g.clone()
        for k, p in enumerate(separate):
            i, s = divmod(k, 2)
            p.grad = g[i, s * h : (s + 1) * h].clone()
        opt_ours.step()
        opt_ref.step()
    for k, p in enumerate(separate):
        i, s = divmod(k, 2)
        torch.testing.assert_close(w13.detach()[i, s * h : (s + 1) * h], p.detach(), rtol=0, atol=2e-3)


def test_upgrade_state_dict_restacks_old_experts():
    """An old per-expert state dict loads into the stacked-expert model with identical weights."""
    model = M.UnifiedModel(tiny_config()).to(DEVICE)
    new = model.state_dict()
    old = {}
    for key, value in new.items():
        if key.endswith(".moe.w13"):
            h = value.shape[1] // 2
            for e in range(value.shape[0]):
                old[key.replace("moe.w13", f"moe.experts.{e}.w1.weight")] = value[e, :h]
                old[key.replace("moe.w13", f"moe.experts.{e}.w3.weight")] = value[e, h:]
        elif key.endswith(".moe.w2"):
            for e in range(value.shape[0]):
                old[key.replace("moe.w2", f"moe.experts.{e}.w2.weight")] = value[e]
        else:
            old[key] = value
    upgraded = M.upgrade_state_dict(old)
    assert upgraded.keys() == new.keys()
    assert all(torch.equal(upgraded[k], new[k]) for k in new)


def _row(n=40, whole=True):
    from brainwm_unified import masking as Mk

    genes = np.arange(n) * 2
    raw = np.r_[np.arange(1, n - 9, dtype=np.float32), np.zeros(9, np.float32)]  # last 9 are measured zeros
    values = np.log1p(raw / raw.sum() * 1e4).astype(np.float32)
    return Mk.Row(genes=genes, values=values, raw=raw, lib=float(raw.sum()), whole_transcriptome=whole)


def test_row_masks_follow_their_definitions():
    from brainwm_unified import masking as Mk

    row, rng = _row(), np.random.default_rng(0)
    states, ev = Mk.RandomGenes(0.15)(row, rng, Mk.Resources())
    assert ev is None and (states == Mk.TARGET).sum() == 6 and not (states == Mk.HIDDEN).any()

    program = np.full(200, -1)
    program[row.genes[:5]], program[row.genes[10:20]] = 0, 1  # two programs present in the row
    states, _ = Mk.GeneProgram(0.15)(row, np.random.default_rng(1), Mk.Resources(gene_program=program))
    picked = set(program[row.genes[states == Mk.TARGET]])
    assert picked <= {0, 1} and all((states[program[row.genes] == p] == Mk.TARGET).all() for p in picked)

    panel = row.genes[::3]
    states, _ = Mk.Panel(target_ratio=0.25, min_visible=4)(row, np.random.default_rng(2), Mk.Resources(panels=[panel]))
    inside = np.isin(row.genes, panel)
    assert (states[inside] == Mk.VISIBLE).all() and (states[~inside] != Mk.VISIBLE).all()
    assert (states == Mk.TARGET).sum() == round(0.25 * (~inside).sum())
    states, _ = Mk.Panel()(_row(whole=False), np.random.default_rng(2), Mk.Resources(panels=[panel]))
    assert not (states == Mk.HIDDEN).any()  # a targeted-panel cell falls back to scattered genes

    states, ev = Mk.Depth(keep=(0.3, 0.3))(row, np.random.default_rng(3), Mk.Resources())
    assert (states == Mk.TARGET).sum() == 6 and ev.shape == row.values.shape
    assert (ev[row.raw == 0] == 0).all() and (ev != row.values).any()  # zeros stay zero; counts get thinned


def test_group_masks_and_curriculum():
    from brainwm_unified import masking as Mk

    rel = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 0, 0], [10, 0, 0], [0, 0, 0]], np.float32)
    valid = np.arange(5)  # member 5 is padding
    m = Mk.Neighbours(0.4)(rel, valid, np.random.default_rng(0))
    assert m.sum() == 2 and not m[5]
    m = Mk.Region(0.4)(rel, valid, np.random.default_rng(0))
    hit = np.flatnonzero(m)
    assert len(hit) == 2 and np.ptp(rel[hit, 0]) <= 2  # two adjacent members, not scattered

    mix = Mk.Mixture(["a", "b"], stages=[(0.2, [1, 0]), (1.01, [0, 1])])
    assert mix.pick(np.random.default_rng(0), 0.1) == "a" and mix.pick(np.random.default_rng(0), 0.5) == "b"
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    assert Mk.Mixture(["only"]).pick(rng, 0.7) == "only" and rng.bit_generator.state == before  # no draw


def test_cajal_v0_batches_are_pinned(subset_dir):
    """cajal_v0 must keep producing the original batches (verified bit-identical to a41a4d6 on unified_1m)."""
    import hashlib

    _, b, _, _ = make(subset_dir)
    digest = hashlib.sha256(b"".join(b[k].cpu().numpy().tobytes() for k in sorted(b))).hexdigest()
    assert digest == "6c2b6cdd712777120036671a0a3069163bd7385a62ca585237c13de75ca6a91e"


def test_table_v1_batches_are_valid(subset_dir, monkeypatch):
    from brainwm_unified import data as D
    from brainwm_unified import masking as Mk

    monkeypatch.setattr(D, "WHOLE_TRANSCRIPTOME", 30)  # synthetic: 50-gene panel = whole, 20-gene panel = targeted
    ds = Subset(subset_dir)
    dcfg = DataConfig(
        root=subset_dir, dissociated_cells=8, spatial_groups=3, neighbors=K, zero_tokens=4, masks="table_v1"
    )
    res = D.mask_resources(ds, dcfg)
    assert len(res.panels) == 1 and len(res.panels[0]) == 20
    res.gene_program = np.arange(N_GENES) % 7
    plan = Mk.preset("table_v1")
    v0 = build_batch(
        ds,
        DataConfig(root=subset_dir, dissociated_cells=8, spatial_groups=3, neighbors=K, zero_tokens=4),
        N_GENES,
        np.random.default_rng(0),
    )
    for progress in (0.1, 0.5, 0.9):
        b = build_batch(ds, dcfg, N_GENES, np.random.default_rng(0), plan=plan, res=res, progress=progress)
        cls, masked, doc = b["tok_cls"], b["tok_masked"], b["tok_doc"]
        assert not (masked & cls).any() and (b["tok_in"][masked] == 0).all()
        assert torch.bincount(doc[masked], minlength=int(cls.sum())).min() >= 1  # every cell has a target
        assert len(b["tok_gene"]) <= len(v0["tok_gene"])  # hidden tokens only ever remove
        assert (b["grp_masked"].sum(1) >= 1).all() and not b["grp_masked"][b["grp_members"] < 0].any()


def test_mask_plans_and_registry_from_toml():
    from brainwm_unified import masking as Mk

    v0 = Mk.load_plan("cajal_v0")
    assert v0.rows.specs == [Mk.RandomGenes(0.15)] and v0.groups.specs == [Mk.Neighbours(0.3)]
    v1 = Mk.load_plan("table_v1")
    assert [type(s).__name__ for s in v1.rows.specs] == ["RandomGenes", "GeneProgram", "Panel", "Depth"]
    with pytest.raises(TypeError, match="valid"):
        Mk.ROW_MASKS.build("random_genes", ratoi=0.2)  # typo names the valid parameters
    with pytest.raises(KeyError, match="registered"):
        Mk.ROW_MASKS.build("no_such_mask")
    with pytest.raises(ValueError, match="weights"):
        Mk.plan_from_dict(
            {
                "rows": {"specs": [{"name": "random_genes"}], "weights": [1, 2]},
                "groups": {"specs": [{"name": "neighbours"}]},
            }
        )


def test_experiments_resolve_extend_and_sweep(tmp_path):
    from brainwm_unified import experiment
    from brainwm_unified.config import TrainConfig

    root = experiment.Path(__file__).resolve().parent.parent / "configs" / "experiments"
    (run,) = experiment.resolve(root / "cajal_1m.toml")
    mcfg, dcfg, tcfg = experiment.build(run)
    assert (tcfg.steps, dcfg.masks, mcfg.n_heads, run["run"]["out"]) == (30000, "cajal_v0", 4, "runs/cajal_1m")
    assert tcfg.lr == TrainConfig().lr  # untouched fields keep the dataclass defaults
    runs = experiment.resolve(root / "single_task.toml", ["train.steps=100"])
    assert len(runs) == 2 and all(r["train"]["steps"] == 100 for r in runs)
    assert [experiment.build(r)[2].task_weights["cell_type"] for r in runs] == [1.0, 0.0]
    sweep_doc = experiment.load(root / "single_task.toml")
    with pytest.raises(ValueError, match="not pinned"):
        experiment.pin_sweep(sweep_doc, ["train.steps=100"])
    (key,) = sweep_doc["sweep"]
    member = experiment.pin_sweep(sweep_doc, [f"{key}={json.dumps(sweep_doc['sweep'][key][1])}"])
    assert "sweep" not in member and member["train"] == sweep_doc["train"]
    (tmp_path / "a.toml").write_text('extends = ["b.toml"]\n')
    (tmp_path / "b.toml").write_text('extends = ["a.toml"]\n')
    with pytest.raises(ValueError, match="cycle"):
        experiment.load(tmp_path / "a.toml")
    with pytest.raises(ValueError, match="no field"):
        experiment.build({"train": {"stpes": 5}})
    with pytest.raises(ValueError, match="status"):
        experiment.build({"status": "maybe"})
    for path in root.rglob("*.toml"):  # every experiment resolves; each finished one says what it found
        doc = experiment.load(path)
        assert experiment.resolve(path) and (path.stem.startswith("_") or doc.get("outcome")), path


def test_cond_hsic_penalises_batch_within_class_only():
    from brainwm_unified.regularizers import CondHSIC

    torch.manual_seed(0)
    n = 120
    y = torch.arange(n, device=DEVICE) % 3
    g = torch.randint(0, 4, (n,), device=DEVICE)
    base = torch.randn(n, 8, device=DEVICE) + 4 * y[:, None].float()  # classes far apart
    hsic = CondHSIC()
    clean = hsic(None, {"cell_type": y, "batch": g}, {"mu": base})
    shifted = hsic(None, {"cell_type": y, "batch": g}, {"mu": base + 3 * g[:, None].float()})
    assert shifted > 5 * clean  # batch shift inside each class is penalised
    confounded = hsic(None, {"cell_type": y, "batch": y}, {"mu": base})  # each class in its own batch
    assert confounded.abs() < 1e-6  # biology that differs by batch is not pulled together


def test_sigreg_prefers_isotropic_gaussian():
    from brainwm_unified.regularizers import SIGReg

    torch.manual_seed(0)
    sig = SIGReg()
    gauss = sig(None, {}, {"mu": torch.randn(2000, 16, device=DEVICE)})
    clustered = sig(
        None, {}, {"mu": torch.randn(2000, 16, device=DEVICE) * 0.2 + 3 * (torch.rand(2000, 1, device=DEVICE) > 0.5)}
    )
    assert clustered > 10 * gauss


def test_latent_options_none_mog_and_detached_reconstruction(subset_dir):
    from brainwm_unified.objectives import latent_penalty

    _, b, _, _ = make(subset_dir)
    for latent_reg in ("none", "mog"):
        torch.manual_seed(0)
        model = M.UnifiedModel(tiny_config(latent_reg=latent_reg, residual_dim=4, recon_detach_mu=True)).to(DEVICE)
        out = model(b, M.dense_mask(b["tok_doc"]), loops=2)
        if latent_reg == "none":
            assert torch.equal(out["z"], out["mu"])
        pen = latent_penalty(model, out)
        assert torch.isfinite(pen)
        if latent_reg == "mog":
            pen.backward(retain_graph=True)
            assert model.prior.means.grad is not None
        model.zero_grad()
        recon_in = model.recon_input(out)
        model.decoder(recon_in, b["batch"]).sum().backward()
        assert model.to_latent.weight.grad is None or model.to_latent.weight.grad.abs().sum() == 0  # mu is cut off
        assert model.res_proj.weight.grad.abs().sum() > 0  # the residual path learns instead (zero-init projection)


def test_task_schedule_phases_then_task_weights():
    from brainwm_unified.config import TrainConfig
    from brainwm_unified.train import task_weights_at

    cfg = TrainConfig()
    assert task_weights_at(cfg, 0) == cfg.task_weights  # no schedule: one set of weights
    a, b = {"cell_type": 1.0}, {"cell_type": 2.0}
    cfg.task_schedule = [{"until": 10, "task_weights": a}, {"until": 20, "task_weights": b}]
    assert [task_weights_at(cfg, s) for s in (0, 9, 10, 19, 20)] == [a, a, b, b, cfg.task_weights]


def test_spatial_residual_mean_starts_at_the_neighbours_mean():
    """On, a fresh neighbourhood model predicts the visible members' mean latent; off, it is unchanged."""
    torch.manual_seed(0)
    z = torch.randn(3, 5, 8)
    rel = torch.randn(3, 5, 3)
    masked = torch.zeros(3, 5, dtype=torch.bool)
    masked[:, 0] = True
    pad = torch.zeros(3, 5, dtype=torch.bool)
    pad[1, 3:] = True
    on = M.NeighborhoodModel(tiny_config(spatial_residual_mean=True, z_dim=8)).eval()
    seen = (~masked & ~pad).unsqueeze(-1).float()
    mean = (z * seen).sum(1) / seen.sum(1)
    torch.testing.assert_close(on(z, rel, masked, pad)[:, 0], mean)
    torch.manual_seed(0)
    off = M.NeighborhoodModel(tiny_config(z_dim=8)).eval()
    assert off.out.weight.abs().sum() > 0 and not torch.allclose(off(z, rel, masked, pad)[:, 0], mean)


def test_gene_rates_use_the_cells_that_measure_each_gene(subset_dir):
    ds = Subset(subset_dir)
    rates = ds.gene_rates(N_GENES)
    lib = np.array([ds.values[ds.row_ptr[c] : ds.row_ptr[c + 1]].sum(dtype=np.float64) for c in range(ds.n_cells)])
    for g in range(N_GENES):
        measures = np.array([g in ds.panel_genes(p) for p in ds.panel_id])
        hit = ds.gene_index == g
        expect = (
            ds.values[hit].sum(dtype=np.float64) / lib[measures].sum()
            if measures.any() and lib[measures].sum()
            else 0.0
        )
        assert rates[g] == pytest.approx(expect, rel=1e-9, abs=1e-12)


def test_pearson_input_residuals(subset_dir):
    """input_norm="pearson" feeds (x - n p) / sqrt(mu + mu^2/100) of the cell's counts; log1p keeps no buffer."""
    ds, b, _, _ = make(subset_dir)
    assert "encoder.gene_rate" not in M.UnifiedModel(tiny_config()).state_dict()  # the default is unchanged
    model = M.UnifiedModel(tiny_config(input_norm="pearson")).to(DEVICE)
    model.encoder.gene_rate[:N_GENES] = torch.from_numpy(ds.gene_rates(N_GENES)).to(DEVICE)
    model.encoder.residual_clip.fill_(1e6)
    r = model.encoder.pearson_residuals(b).double().cpu().numpy()
    ptr, val = b["cnt_ptr"].cpu().numpy(), b["cnt_val"].cpu().numpy().astype(np.float64)
    lib = np.add.reduceat(val, ptr[:-1])
    doc, gene = b["tok_doc"].cpu().numpy(), b["tok_gene"].cpu().numpy()
    tok = ~b["tok_cls"].cpu().numpy()
    n = lib[doc[tok]]
    x = np.expm1(b["tok_in"].double().cpu().numpy()[tok]) * n / 1e4
    visible = ~b["tok_masked"].cpu().numpy()[tok]
    assert (np.abs(x[visible] - np.round(x[visible])) <= 2e-2 * np.maximum(x[visible], 1)).all()  # back to counts
    mu = n * ds.gene_rates(N_GENES)[gene[tok]]
    sd = np.sqrt(mu + mu**2 / 100)
    expect = np.where(sd > 0, (x - mu) / np.where(sd > 0, sd, 1), 0)
    np.testing.assert_allclose(r[tok], expect, rtol=1e-3, atol=1e-3)
    assert (r[tok][visible & (x == 0) & (mu > 0)] < 0).all()  # a measured zero is below expectation
    with pytest.raises(ValueError, match="valid"):
        M.UnifiedModel(tiny_config(input_norm="sct"))


def test_batch_adversary_reverses_the_encoder_gradient(subset_dir):
    """The discriminator descends the batch loss; mu receives the negated gradient; off, nothing is built."""
    from brainwm_unified.regularizers import OBJECTIVES

    _, b, _, _ = make(subset_dir)
    assert not any("batch_adversary" in k for k in M.UnifiedModel(tiny_config()).state_dict())
    with pytest.raises(ValueError, match="batch_adversary"):
        OBJECTIVES.build("batch_adversary")(M.UnifiedModel(tiny_config()).to(DEVICE), b, {"mu": torch.zeros(1, 8)})
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(batch_adversary=16)).to(DEVICE)
    mu = torch.randn(int(b["tok_cls"].sum()), 8, device=DEVICE, requires_grad=True)
    loss = OBJECTIVES.build("batch_adversary")(model, b, {"mu": mu})
    g_rev, *g_disc = torch.autograd.grad(loss, [mu, *model.batch_adversary.parameters()])
    plain = torch.nn.functional.cross_entropy(model.batch_adversary.net(mu), b["batch"])
    g_plain, *g_disc_plain = torch.autograd.grad(plain, [mu, *model.batch_adversary.parameters()])
    torch.testing.assert_close(g_rev, -g_plain)  # the encoder side climbs the discriminator's loss
    for a, p in zip(g_disc, g_disc_plain):  # the discriminator itself descends it
        torch.testing.assert_close(a, p)
