import copy

import pytest
import torch

from brainwm.foundation_moe import (
    BatchedSwiGLUExperts,BrainWMFoundationMoE,DeepSpeedEasyMoEBlock,SwiGLU,
)
from brainwm.sparse_tokens import SparseTokenBatch


def make_experts(count=3, width=8, expansion=2.0):
    from deepspeed.moe.experts import Experts

    return Experts(SwiGLU(width, expansion), count, "test_experts")


def test_batched_experts_preserve_state_keys_shapes_and_parameter_order():
    legacy = make_experts()
    batched = BatchedSwiGLUExperts(copy.deepcopy(legacy), "batched")

    legacy_state = legacy.state_dict()
    batched_state = batched.state_dict()
    assert list(legacy_state) == list(batched_state)
    assert [tuple(value.shape) for value in legacy_state.values()] == [
        tuple(value.shape) for value in batched_state.values()
    ]
    assert [name for name, _ in legacy.named_parameters()] == [
        name for name, _ in batched.named_parameters()
    ]


def test_batched_experts_match_loop_forward_and_gradients_on_cpu():
    torch.manual_seed(17)
    loop = BatchedSwiGLUExperts(make_experts(), "loop")
    batched = BatchedSwiGLUExperts(make_experts(), "batched")
    batched.load_state_dict(loop.state_dict())
    loop_input = torch.randn(1, 3, 5, 8, requires_grad=True)
    batched_input = loop_input.detach().clone().requires_grad_(True)

    loop_output = loop(loop_input)
    batched_output = batched(batched_input)
    torch.testing.assert_close(loop_output, batched_output)

    loop_output.square().sum().backward()
    batched_output.square().sum().backward()
    torch.testing.assert_close(loop_input.grad, batched_input.grad)
    loop_gradients = dict(loop.named_parameters())
    batched_gradients = dict(batched.named_parameters())
    assert loop_gradients.keys() == batched_gradients.keys()
    for name in loop_gradients:
        torch.testing.assert_close(
            loop_gradients[name].grad, batched_gradients[name].grad,
        )


def test_prepared_weights_are_reused_and_cleared_without_changing_gradients():
    torch.manual_seed(23)
    uncached = BatchedSwiGLUExperts(make_experts(), "batched")
    cached = BatchedSwiGLUExperts(make_experts(), "batched")
    cached.load_state_dict(uncached.state_dict())
    uncached_input = torch.randn(1, 3, 5, 8, requires_grad=True)
    cached_input = uncached_input.detach().clone().requires_grad_(True)

    uncached_output = uncached(uncached_input) + uncached(uncached_input)
    cached.prepare_weights()
    prepared = cached._cached_weights
    cached_output = cached(cached_input) + cached(cached_input)
    assert cached._cached_weights is prepared
    cached.clear_prepared_weights()
    assert cached._cached_weights is None

    torch.testing.assert_close(uncached_output, cached_output)
    uncached_output.square().sum().backward()
    cached_output.square().sum().backward()
    torch.testing.assert_close(uncached_input.grad, cached_input.grad)
    for uncached_parameter, cached_parameter in zip(
        uncached.parameters(), cached.parameters(), strict=True,
    ):
        torch.testing.assert_close(uncached_parameter.grad, cached_parameter.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_experts_match_loop_under_bf16_autocast():
    torch.manual_seed(29)
    loop = BatchedSwiGLUExperts(make_experts(count=12, width=64, expansion=2.67), "loop").cuda()
    batched = BatchedSwiGLUExperts(make_experts(count=12, width=64, expansion=2.67), "batched").cuda()
    batched.load_state_dict(loop.state_dict())
    loop_input = torch.randn(1, 12, 32, 64, device="cuda", requires_grad=True)
    batched_input = loop_input.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loop_output = loop(loop_input)
        batched_output = batched(batched_input)
    assert loop_output.dtype == batched_output.dtype == torch.bfloat16
    torch.testing.assert_close(loop_output, batched_output, atol=8e-3, rtol=8e-2)

    loop_output.float().square().sum().backward()
    batched_output.float().square().sum().backward()
    torch.testing.assert_close(loop_input.grad, batched_input.grad, atol=3e-2, rtol=8e-2)
    for (loop_name, loop_parameter), (batched_name, batched_parameter) in zip(
        loop.named_parameters(), batched.named_parameters(), strict=True,
    ):
        assert loop_name == batched_name
        torch.testing.assert_close(
            loop_parameter.grad, batched_parameter.grad, atol=5e-1, rtol=1e-1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("top_k", [1, 2])
def test_grouped_moe_matches_deepspeed_dispatch_and_gradients(top_k):
    """The grouped adapter must preserve the routed block contract."""
    torch.manual_seed(43 + top_k)
    reference=DeepSpeedEasyMoEBlock(
        d_model=64,d_hidden=256,n_experts=12,top_k=top_k,
        attention_heads=4,expert_execution="batched",
        attention_projection_execution="fused_triplet",
    ).cuda()
    grouped=copy.deepcopy(reference)
    grouped.set_expert_execution("grouped")
    reference.train();grouped.train()
    x=torch.randn(32,9,64,device="cuda",requires_grad=True)
    gx=x.detach().clone().requires_grad_(True)
    mask=torch.arange(9,device="cuda").unsqueeze(0)<torch.tensor(
        [9,8,7,6,5,4,3,2,1]+[9]*23,device="cuda",
    ).unsqueeze(1)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        expected,_=reference(x,mask)
        actual,_=grouped(gx,mask)
    torch.testing.assert_close(actual,expected,atol=2e-2,rtol=8e-2)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(x.grad,gx.grad,atol=5e-2,rtol=1e-1)
    ref_params=dict(reference.named_parameters())
    grouped_params=dict(grouped.named_parameters())
    assert ref_params.keys()==grouped_params.keys()
    for name in ref_params:
        torch.testing.assert_close(
            ref_params[name].grad,grouped_params[name].grad,
            atol=7e-1,rtol=2e-1,msg=f"gradient mismatch: {name}",
        )


def _packed_batch(lengths,offset=0):
    lengths=torch.tensor(lengths,device="cuda",dtype=torch.long)
    row_ptr=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)))
    tokens=int(row_ptr[-1])
    return SparseTokenBatch(
        gene_index=(torch.arange(tokens,device="cuda")+offset)%16,
        values=torch.linspace(.1,1.0,tokens,device="cuda"),
        row_ptr=row_ptr,
        measured_gene_index=torch.arange(16,device="cuda"),
        token_mask=torch.ones(tokens,device="cuda",dtype=torch.bool),
        packed=True,packed_layout_validated=True,
        packed_min_seqlen=int(lengths.min()),
        packed_max_seqlen=int(lengths.max()),
        neighbor_edges_present=False,
        row_ids=torch.repeat_interleave(
            torch.arange(len(lengths),device="cuda"),lengths,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_joint_main_source_flash_attention_matches_separate_gradients():
    torch.manual_seed(47)
    separate=BrainWMFoundationMoE(
        16,d_model=8,d_hidden=8,n_blocks=2,n_experts=2,top_k=1,
        attention_heads=2,moe_backend="easy_deepspeed",
        packed_attention_backend="flash_bla",
        attention_projection_execution="fused_triplet",
    ).cuda().train()
    fused=copy.deepcopy(separate)
    main=_packed_batch([3,2])
    source=_packed_batch([2,4],offset=3)

    def run(model,use_fused):
        model.zero_grad(set_to_none=True)
        with model._prepared_backbone_weights():
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                if use_fused:
                    main_output,source_output=model._forward_sparse_flash_pair(
                        main,source,
                    )
                else:
                    main_output=model.forward_sparse(main)
                    source_output=model.forward_sparse(source,encode_only=True)
                loss=(main_output["expression_reconstruction"].float().square().mean()
                      +main_output["completed_expression"].float().square().mean()
                      +main_output["expert_balance_loss"].float()
                      +source_output["cell_embedding"].float().square().mean())
        loss.backward()
        return main_output,source_output,loss

    expected_main,expected_source,expected_loss=run(separate,False)
    actual_main,actual_source,actual_loss=run(fused,True)
    assert separate.expression.null.grad is not None
    assert fused.expression.null.grad is not None
    assert torch.count_nonzero(separate.expression.null.grad)==0
    assert torch.count_nonzero(fused.expression.null.grad)==0
    torch.testing.assert_close(actual_loss,expected_loss,atol=2e-3,rtol=2e-3)
    for name in ("tokens","cell_embedding","expression_reconstruction",
                 "completed_expression","forecast","expert_balance_loss"):
        torch.testing.assert_close(
            actual_main[name],expected_main[name],atol=2e-2,rtol=5e-2,
        )
    torch.testing.assert_close(
        actual_source["cell_embedding"],expected_source["cell_embedding"],
        atol=2e-2,rtol=5e-2,
    )
    separate_parameters=dict(separate.named_parameters())
    fused_parameters=dict(fused.named_parameters())
    assert separate_parameters.keys()==fused_parameters.keys()
    for name in separate_parameters:
        expected=separate_parameters[name].grad
        actual=fused_parameters[name].grad
        assert (expected is None)==(actual is None),name
        if expected is not None:
            torch.testing.assert_close(
                actual,expected,atol=1e-2,rtol=2e-2,
                msg=f"gradient mismatch: {name}",
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_joint_triple_flash_attention_matches_pair_and_keeps_target_inference_only():
    """The optional target fusion must preserve trainable-stream numerics."""
    torch.manual_seed(53)
    pair = BrainWMFoundationMoE(
        16, d_model=8, d_hidden=8, n_blocks=1, n_experts=2, top_k=1,
        attention_heads=2, moe_backend="easy_deepspeed",
        packed_attention_backend="flash_bla",
        attention_projection_execution="fused_triplet",
        fuse_development_attention=True,
        fuse_development_target_attention=False,
    ).cuda().train()
    triple = copy.deepcopy(pair)
    triple.set_fuse_development_target_attention(True)
    main = _packed_batch([3, 2])
    source = _packed_batch([2, 4], offset=3)
    target = _packed_batch([2, 3], offset=5)

    def run(model, use_triple):
        model.zero_grad(set_to_none=True)
        with model._prepared_backbone_weights():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if use_triple:
                    main_output, source_output, target_output = (
                        model._forward_sparse_flash_triple(main, source, target)
                    )
                else:
                    main_output, source_output = model._forward_sparse_flash_pair(
                        main, source,
                    )
                    with torch.inference_mode():
                        target_output = model.forward_sparse(target, encode_only=True)
                loss = (
                    main_output["expression_reconstruction"].float().square().mean()
                    + main_output["completed_expression"].float().square().mean()
                    + main_output["expert_balance_loss"].float()
                    + source_output["cell_embedding"].float().square().mean()
                )
            loss.backward()
        return main_output, source_output, target_output, loss

    expected_main, expected_source, expected_target, expected_loss = run(pair, False)
    actual_main, actual_source, actual_target, actual_loss = run(triple, True)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-3, rtol=2e-3)
    for name in (
        "tokens", "cell_embedding", "expression_reconstruction",
        "completed_expression", "forecast", "expert_balance_loss",
    ):
        torch.testing.assert_close(
            actual_main[name], expected_main[name], atol=2e-2, rtol=5e-2,
        )
    torch.testing.assert_close(
        actual_source["cell_embedding"], expected_source["cell_embedding"],
        atol=2e-2, rtol=5e-2,
    )
    torch.testing.assert_close(
        actual_target["cell_embedding"], expected_target["cell_embedding"],
        atol=2e-2, rtol=5e-2,
    )
    pair_parameters = dict(pair.named_parameters())
    triple_parameters = dict(triple.named_parameters())
    assert pair_parameters.keys() == triple_parameters.keys()
    for name in pair_parameters:
        expected = pair_parameters[name].grad
        actual = triple_parameters[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            torch.testing.assert_close(
                actual, expected, atol=2e-2, rtol=5e-2,
                msg=f"target fusion changed gradient: {name}",
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triple_finish_top2_falls_back_to_population_local_dispatch():
    torch.manual_seed(59)
    block = DeepSpeedEasyMoEBlock(
        d_model=16, d_hidden=32, n_experts=4, top_k=2,
        attention_heads=2, expert_execution="batched",
    ).cuda().train()
    populations = (
        torch.randn(3, 1, 16, device="cuda"),
        torch.randn(2, 1, 16, device="cuda"),
        torch.randn(4, 1, 16, device="cuda"),
    )
    outputs, balances = block._run_moe_many(populations)
    for population, output in zip(populations, outputs, strict=True):
        expected, expected_balance, _ = block._run_moe(population)
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=5e-2)
        torch.testing.assert_close(balances[0], expected_balance, atol=2e-3, rtol=2e-3)
        break
