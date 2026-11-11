from types import SimpleNamespace

import pytest
import torch

from brainwm.multitask_objective import (
    ANATOMY_MODALITIES,
    BRAINWM_LOSS_NAMES,
    CELL_TYPE_MODALITIES,
    MultiObjectiveBalancer,
    brainwm_task_losses,
    structured_modality_mask,
)


def test_loss_telemetry_can_be_deferred_without_changing_objective():
    eager=MultiObjectiveBalancer(("a","b"),mode="fixed")
    deferred=MultiObjectiveBalancer(("a","b"),mode="fixed")
    eager_values={"a":torch.tensor(2.,requires_grad=True),
                  "b":torch.tensor(3.,requires_grad=True)}
    deferred_values={name:value.detach().clone().requires_grad_(True)
                     for name,value in eager_values.items()}

    eager_total,eager_info=eager(eager_values)
    deferred_total,deferred_info=deferred(
        deferred_values,materialize_telemetry=False,
    )
    assert all(isinstance(values["raw"],torch.Tensor)
               for values in deferred_info.values())
    assert deferred.materialize_telemetry(deferred_info)==eager_info
    torch.testing.assert_close(deferred_total,eager_total)

    eager_total.backward();deferred_total.backward()
    for name in eager_values:
        torch.testing.assert_close(
            deferred_values[name].grad,eager_values[name].grad,
        )


def test_loss_balancer_rejects_an_entirely_inactive_objective():
    balancer=MultiObjectiveBalancer(("a","b"),mode="ema_equal")
    with pytest.raises(ValueError,match="no active differentiable loss"):
        balancer({"a":None,"b":None})


def test_loss_balancer_preserves_ema_and_detached_max_scaling():
    ema=MultiObjectiveBalancer(("a","b","c"),mode="ema_equal",decay=.5)
    first={"a":torch.tensor(2.,requires_grad=True),"b":None,
           "c":torch.tensor(8.,requires_grad=True)}
    total,info=ema(first)
    torch.testing.assert_close(total,torch.tensor(5.))
    assert info["a"]["scale"]==pytest.approx(2.5)
    assert info["c"]["scale"]==pytest.approx(.625)
    torch.testing.assert_close(ema.ema,torch.tensor([2.,1.,8.]))
    assert ema.seen.tolist()==[True,False,True]

    second={"a":torch.tensor(4.,requires_grad=True),"b":None,
            "c":torch.tensor(4.,requires_grad=True)}
    total,info=ema(second)
    torch.testing.assert_close(total,torch.tensor(4.5))
    assert info["a"]["scale"]==pytest.approx(1.5)
    assert info["c"]["scale"]==pytest.approx(.75)

    maximum=MultiObjectiveBalancer(("a","b","c"),mode="detached_max_equal")
    total,info=maximum(first)
    torch.testing.assert_close(total,torch.tensor(8.))
    assert info["a"]["scale"]==pytest.approx(4.)
    assert info["c"]["scale"]==pytest.approx(1.)


def test_target_aliases_are_masked_as_one_group():
    observed = {
        name: torch.ones(32, dtype=torch.bool)
        for name in CELL_TYPE_MODALITIES | ANATOMY_MODALITIES | {"donor"}
    }
    masks = structured_modality_mask(
        observed,
        task_probability=0.5,
        all_context_probability=0.0,
        context_probability=0.0,
        generator=torch.Generator().manual_seed(7),
    )
    for group in (CELL_TYPE_MODALITIES, ANATOMY_MODALITIES):
        reference = masks[next(iter(group))]
        assert reference.any()
        assert all(torch.equal(reference, masks[name]) for name in group)
    assert not masks["donor"].any()


def test_all_context_rows_mask_every_observed_modality_only():
    observed = {
        "cell_type": torch.tensor([True, True, False]),
        "cl_id": torch.tensor([True, False, False]),
        "anatomy": torch.tensor([True, True, True]),
        "donor": torch.tensor([True, False, True]),
    }
    masks = structured_modality_mask(
        observed,
        task_probability=0.0,
        all_context_probability=1.0,
        context_probability=0.0,
    )
    for name in observed:
        assert torch.equal(masks[name], observed[name])


def test_sparse_task_anchor_gets_at_least_one_masked_row():
    observed = {
        "cell_type": torch.tensor([False, False, True, False]),
        "cl_id": torch.tensor([False, False, True, False]),
        "anatomy": torch.ones(4, dtype=torch.bool),
    }
    masks = structured_modality_mask(
        observed,
        task_probability=0.0,
        all_context_probability=0.0,
        context_probability=0.0,
    )
    assert masks["cell_type"].tolist() == [False, False, True, False]
    assert masks["cl_id"].tolist() == [False, False, True, False]
    assert masks["anatomy"].sum() == 1


def test_fixed_shape_task_losses_match_indexed_reductions_and_activity():
    reconstruction=torch.tensor([1.0,0.5,4.0,-1.0],requires_grad=True)
    completion=torch.tensor([1.5,0.0,2.5,1.0],requires_grad=True)
    cell_logits=torch.tensor([[2.0,-1.0],[-1.0,2.0]],requires_grad=True)
    batch=SimpleNamespace(
        values=torch.tensor([2.0,0.0,3.0,0.0]),
        token_mask=torch.ones(4,dtype=torch.bool),
        expression_masked=torch.tensor([True,True,False,True]),
        token_sampling_weight=torch.tensor([1.0,4.0,1.0,2.0]),
        cell_type_target=torch.tensor([-1,1]),
        anatomy_target=None,
        modality_masked={"cell_type":torch.tensor([False,True])},
        neighbor_indices=None,
        neighbor_mask=None,
        developmental_observed=None,
    )
    output={
        "expression_reconstruction":reconstruction,
        "completed_expression":completion,
        "canonical_annotation_logits":cell_logits,
        "expert_balance_loss":reconstruction.sum()*0+0.25,
    }

    losses,activity=brainwm_task_losses(
        output,batch,return_activity=True,
    )

    torch.testing.assert_close(losses["mgm"],(reconstruction[0]-2).square())
    selected=torch.tensor([1,3])
    torch.testing.assert_close(
        losses["zero"],
        (reconstruction[selected].square()*torch.tensor([4.,2.])).mean(),
    )
    selected=torch.tensor([0,1,3])
    torch.testing.assert_close(
        losses["completion"],
        torch.nn.functional.smooth_l1_loss(
            completion[selected],batch.values[selected],
        ),
    )
    torch.testing.assert_close(
        losses["cell"],
        torch.nn.functional.cross_entropy(cell_logits[1:],torch.tensor([1])),
    )
    assert bool(activity["mgm"])
    assert bool(activity["zero"])
    assert bool(activity["completion"])
    assert bool(activity["cell"])
    assert not bool(activity["anatomy"])

    balancer=MultiObjectiveBalancer(BRAINWM_LOSS_NAMES,mode="fixed")
    total,_=balancer(losses,active=activity,materialize_telemetry=False)
    total.backward()
    assert reconstruction.grad is not None and torch.isfinite(reconstruction.grad).all()
    assert completion.grad is not None and torch.isfinite(completion.grad).all()
    assert cell_logits.grad is not None and torch.isfinite(cell_logits.grad).all()
