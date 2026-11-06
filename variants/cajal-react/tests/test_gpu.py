"""GPU checks: FlexAttention block mask agrees with the dense reference mask."""

import math

import numpy as np
import pytest
import torch
from test_unified import make, subset_dir, tiny_config  # noqa: F401  (fixture re-export)

from brainwm_unified import model as M
from brainwm_unified.config import DataConfig
from brainwm_unified.data import Subset

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("compiled", [False, True])
def test_flex_matches_dense(subset_dir, compiled):  # noqa: F811
    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(d_model=64)).cuda().eval()  # compiled flex needs head dim >= 16
    if compiled:
        M.enable_compiled_flex()
    with torch.no_grad():
        flex = model(b, M.block_mask(b["tok_doc"]))["mu"]
        dense = model(b, M.dense_mask(b["tok_doc"]))["mu"]
    torch.testing.assert_close(flex, dense, rtol=2e-3, atol=2e-3)


def _has_deepgemm():
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize(
    "backend",
    ["torch", pytest.param("deepgemm", marks=pytest.mark.skipif(not _has_deepgemm(), reason="DeepGEMM not built"))],
)
def test_grouped_backends_agree(backend):
    from brainwm_unified import deepinfra

    torch.manual_seed(0)
    n_experts, d, h = 8, 128, 256
    counts = torch.tensor([5, 0, 17, 3, 40, 1, 0, 9], device="cuda")
    x = torch.randn(int(counts.sum()), d, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(n_experts, o, i, device="cuda", dtype=torch.bfloat16) * 0.1 for o, i in ((h, d), (h, d), (d, h))]
    outs, grads = {}, {}
    for name in ("loop", backend):
        xs = x.clone().requires_grad_()
        wg = [w.clone().requires_grad_() for w in ws]
        y = deepinfra.grouped_swiglu(xs, *wg, counts, name)
        y.float().square().sum().backward()
        outs[name], grads[name] = y.float(), [xs.grad.float()] + [w.grad.float() for w in wg]
    torch.testing.assert_close(outs[backend], outs["loop"], rtol=2e-2, atol=2e-2)
    for a, b in zip(grads[backend], grads["loop"]):
        torch.testing.assert_close(a, b, rtol=5e-2, atol=5e-2)


def test_block_mask_matches_reference_builder():
    """Our boundary-derived BlockMask agrees with FlexAttention's own builder, uneven cells included."""
    from torch.nn.attention.flex_attention import create_block_mask

    lengths = [1, 300, 5, 129, 128, 700, 2]
    doc = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).cuda()
    doc = torch.cat([doc, torch.full(((-len(doc)) % 128,), -1, device="cuda")])
    ours = M.block_mask(doc)
    ref = create_block_mask(lambda b, h, q, kv: doc[q] == doc[kv], None, None, len(doc), len(doc), device="cuda")
    # Every block the reference needs is covered by ours (ours may add blocks that the mask_mod zeroes).
    assert torch.equal(ours.to_dense() | ref.to_dense(), ours.to_dense())
    q = torch.randn(1, 2, len(doc), 16, device="cuda")
    out_ours = M.flex_attention(q, q, q, block_mask=ours)
    out_ref = M.flex_attention(q, q, q, block_mask=ref)
    torch.testing.assert_close(out_ours, out_ref)


def _has_fa3():
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize(
    "backend", ["jagged", pytest.param("fa3", marks=pytest.mark.skipif(not _has_fa3(), reason="FA3 not built"))]
)
def test_varlen_attention_matches_flex(subset_dir, selective, backend):  # noqa: F811
    """Varlen attention (SDPA or FA3) gives FlexAttention's latent and gradients, with and without selective recompute."""
    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    results = {}
    for name in ("flex", backend):
        torch.manual_seed(0)
        cfg = tiny_config(d_model=128, n_heads=2, selective_checkpointing=selective, attention_backend=name)
        model = M.UnifiedModel(cfg).cuda().train()
        layout = M.attention_layout(b["tok_doc"], name)
        torch.manual_seed(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):  # as in training; FA3 only runs in bf16
            mu = model(b, layout, loops=2)["mu"]
        mu.float().square().sum().backward()
        results[name] = (mu.detach().float(), model.encoder.prelude[0].attn.qkv.weight.grad.clone())
    # Tolerances at bf16 resolution: kernels differ in accumulation order, not in the computation.
    torch.testing.assert_close(results[backend][0], results["flex"][0], rtol=2e-2, atol=1e-2)
    grad, ref = results[backend][1], results["flex"][1]
    assert (grad - ref).norm() / ref.norm() < 1e-2  # a few bf16 elements differ; the gradient as a whole agrees


@pytest.mark.parametrize(
    "backend",
    ["torch", pytest.param("deepgemm", marks=pytest.mark.skipif(not _has_deepgemm(), reason="DeepGEMM not built"))],
)
def test_fused_grouped_swiglu_backends_agree(backend):
    from brainwm_unified import deepinfra

    torch.manual_seed(0)
    n_experts, d, h = 8, 128, 256
    counts = torch.tensor([5, 0, 17, 3, 40, 1, 0, 9], device="cuda")
    x = torch.randn(int(counts.sum()), d, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(n_experts, 2 * h, d, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(n_experts, d, h, device="cuda", dtype=torch.bfloat16) * 0.1
    outs = {}
    for name in ("loop", backend):
        outs[name] = deepinfra.grouped_swiglu_fused(x, w13, w2, counts, name).float()
    torch.testing.assert_close(outs[backend], outs["loop"], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_compiled_blocks_match_eager(subset_dir):  # noqa: F811
    """Regional compile with the FA3 backend (nested jagged tensors do not compile yet)."""
    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    results = {}
    for compiled in (False, True):
        torch.manual_seed(0)
        cfg = tiny_config(d_model=128, n_heads=2, attention_backend="fa3", compile_blocks=compiled)
        model = M.UnifiedModel(cfg).cuda().train()
        torch.manual_seed(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(b, M.attention_layout(b["tok_doc"], "fa3"), loops=2)["mu"]
        mu.float().square().sum().backward()
        results[compiled] = (mu.detach().float(), model.encoder.prelude[0].attn.qkv.weight.grad.clone())
    torch.testing.assert_close(results[True][0], results[False][0], rtol=2e-2, atol=1e-2)
    grad, ref = results[True][1], results[False][1]
    assert (grad - ref).norm() / ref.norm() < 1e-2


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_moe_recompute_matches_block_recompute(subset_dir):  # noqa: F811
    """Recomputing only the MoE branch gives the same latent and gradients as recomputing whole blocks."""
    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    results = {}
    for granularity in ("block", "moe"):
        torch.manual_seed(0)
        cfg = tiny_config(
            d_model=128, n_heads=2, attention_backend="fa3", recompute_scope="all", recompute_granularity=granularity
        )
        model = M.UnifiedModel(cfg).cuda().train()
        torch.manual_seed(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(b, M.attention_layout(b["tok_doc"], "fa3"), loops=2)["mu"]
        mu.float().square().sum().backward()
        results[granularity] = (mu.detach().float(), model.encoder.core[0].moe.w13.grad.clone())
    torch.testing.assert_close(results["moe"][0], results["block"][0], rtol=0, atol=0)
    grad, ref = results["moe"][1], results["block"][1]
    assert (grad - ref).norm() / ref.norm() < 1e-3


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_predict_matches_manual_forward(subset_dir):  # noqa: F811
    """infer.predict returns, in the order asked, what a manual eval forward of the same batches gives."""
    from brainwm_unified import infer

    ds = Subset(subset_dir)
    dcfg = DataConfig(root=subset_dir, zero_tokens=4)
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(d_model=128, n_heads=2, attention_backend="fa3")).cuda().eval()
    cells = np.random.default_rng(0).permutation(ds.n_cells)
    budget = 200  # several batches
    res = infer.predict(model, ds, cells, dcfg, token_budget=budget, num_workers=0, seed=5)
    batches = infer.EvalBatches(ds, dcfg, model.cfg.n_genes, infer.plan_batches(ds, cells, dcfg, budget), seed=5)
    assert len(batches) > 1
    for b in batches:
        got = b.pop("cells").numpy()
        assert not b["tok_masked"].any()
        out = infer.forward_batch(model, M.pad_tokens({k: v.cuda() for k, v in b.items()}), model.cfg.loops)
        pos = np.searchsorted(cells, got, sorter=np.argsort(cells))
        rows = np.argsort(cells)[pos]
        torch.testing.assert_close(torch.from_numpy(res["mu"][rows]), out["mu"].cpu())
        torch.testing.assert_close(torch.from_numpy(res["anatomy"][-1][rows]), out["anatomy"][-1].cpu())


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_batch_swap_and_teacher_gram_train_the_encoder(subset_dir, tmp_path):  # noqa: F811
    from brainwm_unified.config import to_dict
    from brainwm_unified.regularizers import BatchSwap, TeacherGram

    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    cfg = tiny_config(d_model=128, n_heads=2, attention_backend="fa3", compile_blocks=False)
    torch.manual_seed(0)
    teacher = M.UnifiedModel(cfg)
    ckpt = tmp_path / "teacher.pt"
    torch.save(
        {
            "model": teacher.state_dict(),
            "config": to_dict(
                cfg, DataConfig(), __import__("brainwm_unified.config", fromlist=["TrainConfig"]).TrainConfig()
            ),
            "step": 0,
        },
        ckpt,
    )
    torch.manual_seed(1)
    model = M.UnifiedModel(cfg).cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b, M.attention_layout(b["tok_doc"], "fa3"), loops=2)
        losses = {"swap": BatchSwap(n_cells=4)(model, b, out), "gram": TeacherGram(checkpoint=str(ckpt))(model, b, out)}
    for name, loss in losses.items():
        model.zero_grad()
        assert torch.isfinite(loss), name
        loss.backward(retain_graph=True)
        grad = model.encoder.prelude[0].attn.qkv.weight.grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_grad_probe_covers_extra_objectives(subset_dir):  # noqa: F811
    """The probe reports norms and cosines for every loss it is given, regularisers included."""
    from brainwm_unified.regularizers import SIGReg
    from brainwm_unified.train import grad_probe

    _, b, _, _ = make(subset_dir)
    b = M.pad_tokens({k: v.cuda() for k, v in b.items()})
    model = M.UnifiedModel(tiny_config(d_model=128, n_heads=2, attention_backend="fa3", compile_blocks=False))
    model = model.cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b, M.attention_layout(b["tok_doc"], "fa3"), loops=2)
        losses = {"latent": out["mu"].pow(2).mean(), "sigreg": SIGReg()(model, b, out)}
    probe = grad_probe(model, losses)
    assert probe["norm/sigreg"] > 0 and -1 <= probe["cos/latent~sigreg"] <= 1
    assert "cos/core.0/latent~sigreg" in probe


# A tiny model for end-to-end runs of train.main.
TINY_RUN = [
    "model.d_model=128",
    "model.n_heads=2",
    "model.n_genes=50",
    "model.expert_hidden=32",
    "model.n_experts=4",
    "model.top_k=2",
    "model.z_dim=8",
    "model.decoder_hidden=16",
    "model.spatial_d=16",
    "model.compile_blocks=false",
    "data.dissociated_cells=8",
    "data.spatial_groups=2",
    "data.neighbors=4",
    "data.zero_tokens=4",
    "train.num_workers=0",
    "train.log_every=1",
    "train.grad_probe_every=0",
    "train.warmup_steps=2",
    "train.ckpt_every=1000",
    "train.norm_steps=2",
    "train.keep_every=6",
]


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_stop_at_then_resume_follows_one_schedule(subset_dir, tmp_path):  # noqa: F811
    """Stopping at stop_at and resuming continues the same LR schedule and step count to the end."""
    import json
    import math
    from pathlib import Path

    from brainwm_unified import train
    from brainwm_unified.config import TrainConfig

    common = ["--data", subset_dir, "--steps", "6", "--no-compile"]
    whole, split = str(tmp_path / "whole"), str(tmp_path / "split")
    train.main([*common, "--out", whole, "--set", *TINY_RUN])
    train.main([*common, "--out", split, "--set", *TINY_RUN, "train.stop_at=3"])
    assert json.loads((tmp_path / "split" / "record.json").read_text())["status"] == "done"
    train.main([*common, "--out", split, "--resume", "--set", *TINY_RUN])
    lr = {}
    for run in (whole, split):
        rows = [json.loads(line) for line in Path(run, "train.jsonl").read_text().splitlines()]
        lr[run] = {r["step"]: r["lr"] for r in rows}
    assert sorted(lr[split]) == sorted(lr[whole]) == list(range(6))
    for step, value in lr[whole].items():
        assert math.isclose(lr[split][step], value, rel_tol=1e-9), step
    rec = json.loads((tmp_path / "split" / "record.json").read_text())
    assert (rec["status"], rec["step"]) == ("done", 6)
    kept = torch.load(Path(split, "step6.pt"), map_location="cpu", weights_only=False)  # stop step is kept too
    assert set(kept) == {"model", "step", "config"}  # weights only; latest.pt keeps the optimiser state
    assert TrainConfig().steps  # the schedule length, not stop_at, defines the LR curve


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
@pytest.mark.parametrize("balancer", ["grad_equal", "famo"])
def test_balancers_and_budgets_train_and_resume(subset_dir, tmp_path, balancer):  # noqa: F811
    """A balancer plus a budgeted objective run end to end, log their weights and resume from state."""
    import json
    from pathlib import Path

    import torch as T

    from brainwm_unified import train

    run = str(tmp_path / balancer)
    sets = [
        *TINY_RUN,
        f"train.balancer={json.dumps(balancer)}",
        'train.balancer_params={"every": 2}' if balancer == "grad_equal" else "train.balancer_params={}",
        'train.objectives=[{"name": "sigreg", "weight": 0.5, "budget": {"cell_type": 100.0}}]',
        "train.kl_target=5.0",
        "train.kl_warmup_steps=1",
    ]
    common = ["--data", subset_dir, "--steps", "6", "--no-compile", "--out", run]
    train.main([*common, "--set", *sets, "train.stop_at=3"])
    state = T.load(Path(run, "latest.pt"), weights_only=False)
    assert state["duals"]["sigreg"]["log_weight"] > math.log(0.5)  # far within budget: the weight grew
    train.main([*common, "--resume", "--set", *sets])
    rows = [json.loads(line) for line in Path(run, "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(6))
    assert all("weight/sigreg" in r and "scale/gene" in r for r in rows)
    assert len({r["kl_weight"] for r in rows[1:]}) > 1  # the KL weight follows its capacity target
    assert all(np.isfinite(r["total"]) for r in rows)
    if balancer == "grad_equal":  # past norm_steps every loss is scaled, the extra included
        assert "scale/sigreg" in rows[-1]
    else:
        assert abs(sum(v for k, v in rows[-1].items() if k.startswith("famo/")) - 1) < 1e-5


def _model_and_cfg(subset_dir):  # noqa: F811
    torch.manual_seed(0)
    model = M.UnifiedModel(tiny_config(d_model=128, n_heads=2, attention_backend="fa3", compile_blocks=False))
    return model.cuda().eval(), DataConfig(root=subset_dir, neighbors=4, zero_tokens=4)


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_predict_genes_masks_the_training_way_and_is_reproducible(subset_dir):  # noqa: F811
    from brainwm_unified import infer
    from brainwm_unified.impute_metrics import STATS, row_stats

    ds = Subset(subset_dir)
    model, cfg = _model_and_cfg(subset_dir)
    cells = np.arange(ds.n_cells)[::-1]  # any order
    a = infer.predict_genes(model, ds, cells, cfg, num_workers=0, seed=3)
    b = infer.predict_genes(model, ds, cells, cfg, num_workers=0, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in ("cell", "gene", "target", "pred"))
    assert len(a["ptr"]) == len(cells) + 1 and a["ptr"][-1] == len(a["gene"])
    for i, c in enumerate(cells):  # tokens of cell i in the order given, each a gene its panel measures
        sl = slice(a["ptr"][i], a["ptr"][i + 1])
        assert (a["cell"][sl] == c).all() and np.isin(a["gene"][sl], ds.panel_genes(ds.panel_id[c])).all()
    tokens = sum(min(ds.row_ptr[c + 1] - ds.row_ptr[c], cfg.max_nonzero_tokens) + cfg.zero_tokens for c in cells)
    assert 0.08 < len(a["gene"]) / tokens < 0.25  # the 15% training gene mask
    assert np.isfinite(a["pred"]).all()
    assert set(row_stats(a["pred"], a["target"], a["ptr"])) == set(STATS)


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_predict_spatial_never_sees_the_hidden_cell(subset_dir):  # noqa: F811
    """Changing a centre's own expression leaves its prediction unchanged; its truth is its own profile."""
    from brainwm_unified import infer
    from brainwm_unified.impute_metrics import panel_profile

    ds = Subset(subset_dir)
    model, cfg = _model_and_cfg(subset_dir)
    centers = ds.spatial_centers
    a = infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400)
    c = int(centers[3])
    lo, hi = ds.row_ptr[c], ds.row_ptr[c + 1]
    ds.values[lo:hi] = ds.values[lo:hi][::-1] * 7 + 1  # a different profile for one centre
    b = infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400)
    sl = slice(a["ptr"][3], a["ptr"][4])
    np.testing.assert_allclose(a["pred"][sl], b["pred"][sl], rtol=1e-4, atol=1e-4)
    assert not np.allclose(a["true"][sl], b["true"][sl])
    np.testing.assert_array_equal(b["true"][sl], panel_profile(ds.row_ptr, ds.gene_index, ds.values, c, a["gene"][sl]))
    assert len(a["ptr"]) == len(centers) + 1 and a["cell_type"][-1].shape == (len(centers), 6)
    assert np.isfinite(a["pred"]).all()
    # the expected-log readout uses the neighbours' library, never the hidden cell's, and is seeded
    e1 = infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, readout="expected_log")
    e2 = infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, readout="expected_log")
    np.testing.assert_array_equal(e1["pred"], e2["pred"])
    assert np.isfinite(e1["pred"]).all() and (e1["pred"] >= 0).all()
    assert e1["pred"][sl].mean() < b["pred"][sl].mean()  # E[log] <= log E (Jensen)
    ds.values[lo:hi] = ds.values[lo:hi] * 50  # the hidden cell's own library changes; its prediction does not
    e3 = infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, readout="expected_log")
    np.testing.assert_allclose(e1["pred"][sl], e3["pred"][sl], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_predict_spatial_diagnostic_sources(subset_dir):  # noqa: F811
    """source="self" decodes the centre's own latent (it moves with its profile); "neighbour_mean" does not."""
    from brainwm_unified import infer

    ds = Subset(subset_dir)
    model, cfg = _model_and_cfg(subset_dir)
    centers = ds.spatial_centers
    run = {
        s: infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, source=s)
        for s in ("self", "neighbour_mean")
    }
    c = int(centers[3])
    lo, hi = ds.row_ptr[c], ds.row_ptr[c + 1]
    ds.values[lo:hi] = ds.values[lo:hi][::-1] * 7 + 1
    sl = slice(run["self"]["ptr"][3], run["self"]["ptr"][4])
    after = {s: infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, source=s) for s in run}
    assert not np.allclose(run["self"]["pred"][sl], after["self"]["pred"][sl], atol=1e-4)
    np.testing.assert_allclose(
        run["neighbour_mean"]["pred"][sl], after["neighbour_mean"]["pred"][sl], rtol=1e-4, atol=1e-4
    )
    with pytest.raises(ValueError, match="valid"):
        infer.predict_spatial(model, ds, centers, cfg, num_workers=0, token_budget=400, source="oracle")


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_curriculum_self_teacher(subset_dir, tmp_path):  # noqa: F811
    """A cell-type-only phase, then every task plus a Gram anchor to the run's own checkpoint from that step."""
    import json
    from pathlib import Path

    from brainwm_unified import train

    run = str(tmp_path / "curriculum")
    only_ct = {"gene": 0.0, "integration": 0.0, "cell_type": 1.0, "anatomy": 0.0, "spatial": 0.0}
    sets = [
        *TINY_RUN,
        "train.ckpt_every=3",
        "train.keep_every=3",
        f"train.task_schedule={json.dumps([{'until': 3, 'task_weights': only_ct}])}",
        f"train.objectives={json.dumps([{'name': 'teacher_gram', 'start': 3, 'checkpoint': run + '/step3.pt'}])}",
    ]
    train.main(["--data", subset_dir, "--steps", "6", "--no-compile", "--out", run, "--set", *sets])
    rows = [json.loads(line) for line in Path(run, "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(6))
    assert not any("loss/teacher_gram" in r for r in rows[:3])  # off, not computed, before its start
    assert all("loss/teacher_gram" in r and np.isfinite(r["total"]) for r in rows[3:])
    assert Path(run, "step3.pt").exists()  # the teacher is the run's own step-3 checkpoint
