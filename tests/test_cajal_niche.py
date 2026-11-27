from dataclasses import asdict, replace

import torch

from brainwm_unified.config import ModelConfig
from brainwm_unified.infer import load_model
from brainwm_unified.model import UnifiedModel
from brainwm_unified.regularizers import OBJECTIVES
from nanobrain.infra.niche import NICHE_COMPONENT, composition_targets


def config():
    return ModelConfig(
        n_genes=5,
        n_batches=1,
        d_model=8,
        n_heads=2,
        n_prelude=0,
        n_core=0,
        n_coda=0,
        z_dim=2,
        cell_type_levels=[3],
        anatomy_levels=[2],
        spatial_d=8,
        compile_blocks=False,
    )


def test_opt_in_adds_only_niche_head_and_loads_checkpoint(tmp_path):
    base = config()
    opt_in = replace(base, research_components=[NICHE_COMPONENT])
    plain = UnifiedModel(base)
    model = UnifiedModel(opt_in)
    assert set(model.state_dict()) - set(plain.state_dict()) == {"niche_head.weight", "niche_head.bias"}
    assert not hasattr(plain, "niche_head")
    path = tmp_path / "niche.pt"
    torch.save({"model": model.state_dict(), "config": {"ModelConfig": asdict(opt_in)}}, path)
    loaded, _ = load_model(str(path), device="cpu")
    torch.testing.assert_close(loaded.niche_head.weight, model.niche_head.weight)
    torch.testing.assert_close(loaded.niche_head.bias, model.niche_head.bias)


def test_composition_supervises_only_eligible_centres():
    labels = torch.tensor([2, 0, 1, 1, -1])
    members = torch.tensor([[0, 1, 2, 3], [4, 4, -1, -1], [2, 4, -1, -1]])
    target, eligible = composition_targets(labels, members, 3)
    torch.testing.assert_close(target, torch.tensor([[1 / 3, 2 / 3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert eligible.tolist() == [True, False, False]

    model = UnifiedModel(replace(config(), research_components=[NICHE_COMPONENT]))
    objective = OBJECTIVES.build("niche_composition")
    mu = torch.ones((len(labels), 2), requires_grad=True)
    loss = objective(model, {"cell_type": labels, "grp_members": members}, {"mu": mu})
    expected = -(target[0] * model.niche_head(mu[0]).log_softmax(-1)).sum()
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss, mu)
    assert torch.count_nonzero(gradient[1:]) == 0
    assert torch.count_nonzero(gradient[0]) > 0

    empty_loss = objective(model, {"cell_type": labels, "grp_members": members[1:]}, {"mu": mu})
    assert empty_loss.item() == 0.0
    assert torch.count_nonzero(torch.autograd.grad(empty_loss, mu)[0]) == 0
