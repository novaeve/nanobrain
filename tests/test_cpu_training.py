import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_unified import N_GENES, subset_dir  # noqa: F401

from brainwm_unified import model as M
from brainwm_unified.objectives import TASKS
from brainwm_unified.train import main


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cajal_prediction_and_scoring_forward_device(monkeypatch, device):
    from brainwm_unified import cli

    prediction = []
    monkeypatch.setattr(sys, "argv", sys.argv[:])
    monkeypatch.setattr(sys, "path", sys.path[:])
    monkeypatch.setitem(sys.modules, "predict_eval", SimpleNamespace(main=lambda: prediction.extend(sys.argv)))
    cli.main(["predict", "model.pt", "--out", "prediction.npz", "--device", device])
    assert prediction[-2:] == ["--device", device]

    commands = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    score_args = ["--pred", "candidate=prediction.npz", "--out", "scores"]
    with pytest.raises(SystemExit) as stopped:
        cli.main(["eval", "--device", device, *score_args])
    assert stopped.value.code == 0
    (command,) = commands
    assert command[0] == (sys.executable if device == "cpu" else "bash")
    assert command[-6:] == ["--device", device, *score_args]


def test_cpu_attention_matches_dense_attention_on_real_tokens():
    doc = torch.tensor([0, 0, 1, 1, 1, -1, -1])
    values = torch.randn(len(doc), 8, requires_grad=True)
    attention = M.Attention(8, 2)
    packed = attention(values, M.attention_layout(doc, "cpu"))
    dense = attention(values, M.dense_mask(doc))
    mask = doc >= 0
    torch.testing.assert_close(packed[mask], dense[mask])
    torch.testing.assert_close(
        torch.autograd.grad(packed[mask].sum(), values)[0], torch.autograd.grad(dense[mask].sum(), values)[0]
    )


def test_cpu_training_updates_v3_and_resumes(subset_dir, tmp_path):  # noqa: F811
    args = [
        "--config",
        str(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "cajal_v3.toml"),
        "--data",
        subset_dir,
        "--out",
        str(tmp_path),
        "--device",
        "cpu",
        "--steps",
        "2",
        "--set",
        f"model.n_genes={N_GENES}",
        "model.d_model=16",
        "model.n_heads=2",
        "model.n_prelude=1",
        "model.n_core=1",
        "model.n_coda=1",
        "model.min_loops=1",
        "model.max_loops=1",
        "model.n_experts=4",
        "model.top_k=2",
        'model.moe_backend="torch"',
        "model.expert_hidden=16",
        "model.z_dim=8",
        "model.spatial_d=16",
        "model.spatial_layers=1",
        "data.dissociated_cells=6",
        "data.spatial_groups=2",
        "data.neighbors=4",
        "data.zero_tokens=4",
        "train.num_workers=0",
        "train.ckpt_every=1",
        "train.keep_every=1",
        "train.grad_probe_every=0",
        "train.log_every=1",
    ]
    main([*args, "train.stop_at=1"])
    main([*args, "--resume"])
    first = torch.load(tmp_path / "step1.pt", map_location="cpu", weights_only=True)
    last = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
    assert (first["step"], last["step"]) == (1, 2)
    config = last["config"]["ModelConfig"]
    assert config["attention_backend"] == "fa3" and config["moe_backend"] == "torch" and config["compile_blocks"]
    assert last["kl_dual"] is not None
    assert not torch.equal(first["model"]["impute_gctx.2.weight"], last["model"]["impute_gctx.2.weight"])
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["device"] == "cpu" and record["status"] == "done"
    steps = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in steps] == [0, 1]
    assert all(math.isfinite(r[f"loss/{name}"]) for r in steps for name in (*TASKS, "self_gram"))
