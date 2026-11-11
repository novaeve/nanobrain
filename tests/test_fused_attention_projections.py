import copy

import pytest
import torch

from brainwm.foundation_moe import (
    DifferentialLinearAttention,
    _compiled_differential_merge,
    _compiled_flash_bla_feature_map,
    _differential_merge_eager,
    _flash_bla_feature_map_eager,
)


def make_attention(execution: str):
    return DifferentialLinearAttention(
        d_model=8,
        d_hidden=12,
        n_heads=3,
        projection_execution=execution,
    )


def assert_parameter_gradients_close(reference, candidate, **kwargs):
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert reference_parameters.keys() == candidate_parameters.keys()
    for name in reference_parameters:
        torch.testing.assert_close(
            reference_parameters[name].grad,
            candidate_parameters[name].grad,
            msg=lambda message, name=name: f"{name}: {message}",
            **kwargs,
        )


def test_fused_attention_projection_preserves_parameter_contract():
    separate = make_attention("separate")
    for execution in ("fla_layernorm_qkv", "fused", "fused_triplet"):
        fused = make_attention(execution)
        assert list(separate.state_dict()) == list(fused.state_dict())
        assert [tuple(value.shape) for value in separate.state_dict().values()] == [
            tuple(value.shape) for value in fused.state_dict().values()
        ]
        assert [name for name, _ in separate.named_parameters()] == [
            name for name, _ in fused.named_parameters()
        ]


@pytest.mark.parametrize("execution", ["fla_layernorm_qkv", "fused", "fused_triplet"])
def test_fused_attention_projection_matches_forward_and_backward_on_cpu(execution):
    torch.manual_seed(31)
    separate = make_attention("separate")
    fused = make_attention(execution)
    fused.load_state_dict(separate.state_dict())
    separate_input = torch.randn(4, 7, 8, requires_grad=True)
    fused_input = separate_input.detach().clone().requires_grad_(True)
    mask = torch.arange(7).unsqueeze(0) < torch.tensor([7, 5, 3, 6]).unsqueeze(1)

    separate_output = separate(separate_input, mask)
    fused_output = fused(fused_input, mask)
    torch.testing.assert_close(separate_output, fused_output)

    separate_output.square().sum().backward()
    fused_output.square().sum().backward()
    torch.testing.assert_close(separate_input.grad, fused_input.grad)
    assert_parameter_gradients_close(separate, fused)


@pytest.mark.parametrize("execution", ["fla_layernorm_qkv", "fused", "fused_triplet"])
def test_prepared_attention_projection_is_reused_and_cleared(execution):
    torch.manual_seed(37)
    uncached = make_attention(execution)
    cached = copy.deepcopy(uncached)
    uncached_input = torch.randn(3, 5, 8, requires_grad=True)
    cached_input = uncached_input.detach().clone().requires_grad_(True)

    uncached_output = uncached(uncached_input) + uncached(uncached_input)
    cached.prepare_projection()
    prepared = cached._cached_projection
    cached_output = cached(cached_input) + cached(cached_input)
    assert cached._cached_projection is prepared
    cached.clear_prepared_projection()
    assert cached._cached_projection is None

    torch.testing.assert_close(uncached_output, cached_output)
    uncached_output.square().sum().backward()
    cached_output.square().sum().backward()
    torch.testing.assert_close(uncached_input.grad, cached_input.grad)
    assert_parameter_gradients_close(uncached, cached)


def test_feature_map_reference_is_tensor_only_and_preserves_gradients():
    torch.manual_seed(43)
    queries = torch.randn(11, 4, 6, requires_grad=True)
    keys = torch.randn(11, 4, 6, requires_grad=True)
    lengths = torch.arange(11, 22, dtype=torch.long)
    actual_q, actual_k = _flash_bla_feature_map_eager(
        queries, keys, lengths,
    )
    expected_q = queries.exp().clamp(max=1e4) * lengths.to(
        queries.dtype,
    ).view(-1, 1, 1)
    expected_k = keys.exp().clamp(max=1e4)
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    (actual_q.square().mean() + actual_k.square().mean()).backward()
    assert queries.grad is not None and keys.grad is not None


def test_differential_merge_reference_preserves_equation_and_gradients():
    torch.manual_seed(45)
    positive = torch.randn(7, 3, 4, requires_grad=True)
    correction = torch.randn_like(positive, requires_grad=True)
    parameters = [
        (torch.randn(12) * 0.02).requires_grad_() for _ in range(4)
    ]
    output = _differential_merge_eager(
        positive, correction, *parameters, 0.25,
    )
    expected_lambda = 0.25 + torch.exp((parameters[0] * parameters[1]).sum()) \
        - torch.exp((parameters[2] * parameters[3]).sum())
    torch.testing.assert_close(output, positive - expected_lambda * correction)
    output.square().mean().backward()
    assert positive.grad is not None and correction.grad is not None
    assert all(parameter.grad is not None for parameter in parameters)


def test_feature_map_execution_policy_keeps_parameter_contract_on_cpu():
    reference = DifferentialLinearAttention(
        d_model=8, d_hidden=12, n_heads=3, feature_map_execution="torch",
    )
    candidate = DifferentialLinearAttention(
        d_model=8, d_hidden=12, n_heads=3, feature_map_execution="compile",
    )
    candidate.load_state_dict(reference.state_dict())
    candidate.set_feature_map_min_tokens(1)
    x = torch.randn(2, 5, 8, requires_grad=True)
    output = candidate(x)
    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()
    assert list(reference.state_dict()) == list(candidate.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_feature_map_matches_eager_forward_and_backward():
    torch.manual_seed(47)
    eager_q = torch.randn(
        257, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    eager_k = torch.randn_like(eager_q, requires_grad=True)
    compiled_q = eager_q.detach().clone().requires_grad_(True)
    compiled_k = eager_k.detach().clone().requires_grad_(True)
    lengths = torch.randint(1, 65, (257,), device="cuda")
    expected = _flash_bla_feature_map_eager(eager_q, eager_k, lengths)
    actual = _compiled_flash_bla_feature_map(
        compiled_q, compiled_k, lengths,
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    sum(value.float().square().mean() for value in expected).backward()
    sum(value.float().square().mean() for value in actual).backward()
    torch.testing.assert_close(compiled_q.grad, eager_q.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(compiled_k.grad, eager_k.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_differential_merge_matches_eager_forward_and_backward():
    torch.manual_seed(49)
    eager_positive = torch.randn(
        257, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    eager_correction = torch.randn_like(eager_positive, requires_grad=True)
    eager_parameters = [
        (torch.randn(64, device="cuda") * 0.02).requires_grad_()
        for _ in range(4)
    ]
    compiled_positive = eager_positive.detach().clone().requires_grad_(True)
    compiled_correction = eager_correction.detach().clone().requires_grad_(True)
    compiled_parameters = [
        parameter.detach().clone().requires_grad_(True)
        for parameter in eager_parameters
    ]
    expected = _differential_merge_eager(
        eager_positive, eager_correction, *eager_parameters, 0.5,
    )
    actual = _compiled_differential_merge(
        compiled_positive, compiled_correction, *compiled_parameters, 0.5,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        compiled_positive.grad, eager_positive.grad, atol=2e-3, rtol=2e-2,
    )
    torch.testing.assert_close(
        compiled_correction.grad, eager_correction.grad, atol=2e-3, rtol=2e-2,
    )
    for actual_parameter, expected_parameter in zip(
        compiled_parameters, eager_parameters, strict=True,
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad,
            atol=2e-3, rtol=2e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("execution", ["fla_layernorm_qkv", "fused", "fused_triplet"])
def test_fused_projection_matches_separate_on_flash_bla_bf16(execution):
    torch.manual_seed(41)
    separate = make_attention("separate").cuda()
    fused = make_attention(execution).cuda()
    fused.load_state_dict(separate.state_dict())
    lengths = torch.tensor([3, 5, 2], device="cuda", dtype=torch.long)
    row_ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
    row_ids = torch.repeat_interleave(torch.arange(3, device="cuda"), lengths)
    separate_input = torch.randn(10, 8, device="cuda", requires_grad=True)
    fused_input = separate_input.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        separate_output = separate.forward_varlen(
            separate_input, row_ptr, lengths[row_ids], int(lengths.max()),
        )
        fused_output = fused.forward_varlen(
            fused_input, row_ptr, lengths[row_ids], int(lengths.max()),
        )
    torch.testing.assert_close(separate_output, fused_output, atol=2e-2, rtol=5e-2)

    separate_output.float().square().sum().backward()
    fused_output.float().square().sum().backward()
    torch.testing.assert_close(
        separate_input.grad, fused_input.grad, atol=3e-2, rtol=8e-2,
    )
    assert_parameter_gradients_close(
        separate, fused, atol=5e-1, rtol=1e-1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("execution", ["fused", "fused_triplet"])
def test_prepared_projection_cast_cache_preserves_bf16_gradients(execution):
    """Prepared BF16 views may be reused across streams without changing grads."""
    torch.manual_seed(53)
    reference = make_attention(execution).cuda()
    cached = make_attention(execution).cuda()
    cached.load_state_dict(reference.state_dict())
    reference_input = torch.randn(31, 8, device="cuda", requires_grad=True)
    cached_input = reference_input.detach().clone().requires_grad_(True)
    row_ptr = torch.tensor([0, 11, 23, 31], device="cuda", dtype=torch.int32)
    row_ids = torch.repeat_interleave(
        torch.arange(3, device="cuda"),
        torch.tensor([11, 12, 8], device="cuda"),
    )
    cached.prepare_projection()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_output = reference.forward_varlen(
            reference_input, row_ptr, row_ids.new_tensor([11] * 11 + [12] * 12 + [8] * 8),
            12,
        )
        cached_output = cached.forward_varlen(
            cached_input, row_ptr, row_ids.new_tensor([11] * 11 + [12] * 12 + [8] * 8),
            12,
        )
    torch.testing.assert_close(reference_output, cached_output, atol=3e-2, rtol=5e-2)
    reference_output.float().square().sum().backward()
    cached_output.float().square().sum().backward()
    torch.testing.assert_close(reference_input.grad, cached_input.grad, atol=3e-2, rtol=8e-2)
    assert_parameter_gradients_close(reference, cached, atol=5e-1, rtol=1e-1)
