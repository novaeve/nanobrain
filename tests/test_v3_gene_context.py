from dataclasses import asdict

import torch

from brainwm_unified.config import ModelConfig
from brainwm_unified.infer import load_model
from brainwm_unified.model import UnifiedModel, neighbour_gene_values


def test_v3_neighbour_gene_context_excludes_target_cell():
    batch = {
        "cnt_ptr": torch.tensor([0, 1, 3, 4]),
        "cnt_gene": torch.tensor([0, 0, 1, 0]),
        "cnt_val": torch.tensor([80.0, 10.0, 10.0, 4.0]),
        "tok_doc": torch.tensor([0, 1, 2]),
        "tok_gene": torch.tensor([0, 0, 0]),
        "tok_masked": torch.tensor([True, True, True]),
    }
    means = neighbour_gene_values(batch, torch.tensor([[0, 1, -1]]), n_genes=2)
    expected = torch.log1p(torch.tensor([5_000.0, 10_000.0, 0.0]))
    torch.testing.assert_close(means, expected)


def test_v3_checkpoint_uses_learned_context_bias_without_spatial_neighbours(tmp_path):
    cfg = ModelConfig(
        d_model=8,
        n_heads=2,
        n_genes=6,
        n_batches=1,
        n_prelude=0,
        n_core=0,
        n_coda=0,
        compile_blocks=False,
        impute_context="neighbour_genes",
    )
    original = UnifiedModel(cfg)
    with torch.no_grad():
        original.impute_gctx[2].bias.fill_(1.0)
    path = tmp_path / "v3.pt"
    saved = {**asdict(cfg), "ffn": "moe", "niche_head": False, "latent_prior": {}}
    torch.save({"model": original.state_dict(), "config": {"ModelConfig": saved}}, path)
    model, _ = load_model(str(path), device="cpu")
    masked = torch.tensor([False, True])
    hidden = torch.zeros((2, cfg.d_model))
    predicted = model.impute(hidden, {"tok_masked": masked})
    expected = model.gene_head(hidden[masked] + 1.0).squeeze(-1)
    torch.testing.assert_close(predicted, expected)
