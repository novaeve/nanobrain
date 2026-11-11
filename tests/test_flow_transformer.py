import copy

import pytest
import torch

from brainwm.flow_transformer import FlowMatchingTransformer
from brainwm.foundation_moe import BrainWMFoundationMoE
from brainwm.losses import SinkhornRuntime, sinkhorn_transport_plan


def test_population_flow_transformer_is_permutation_equivariant():
    torch.manual_seed(3)
    model = FlowMatchingTransformer(width=16, heads=4, depth=4).eval()
    context = torch.randn(7, 16)
    state = torch.randn(7, 16)
    times = torch.randn(7, 3)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    direct = model(context, state, times)
    permuted = model(context[permutation], state[permutation], times[permutation])
    assert torch.allclose(permuted, direct[permutation], atol=2e-5, rtol=2e-5)


def test_population_flow_transformer_backpropagates_through_unet_path():
    torch.manual_seed(5)
    model = FlowMatchingTransformer(width=16, heads=4, depth=4)
    prediction = model(torch.randn(6, 16), torch.randn(6, 16), torch.randn(6, 3))
    prediction.square().mean().backward()
    names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert any(name.startswith("encoder") for name in names)
    assert any(name.startswith("middle") for name in names)
    assert any(name.startswith("skip_fusion") for name in names)
    assert any(name.startswith("decoder") for name in names)


def test_linear_population_attention_is_permutation_equivariant():
    torch.manual_seed(7)
    model=FlowMatchingTransformer(width=16,heads=4,depth=2,attention_mode="linear").eval()
    context=torch.randn(19,16);state=torch.randn(19,16);times=torch.randn(19,3)
    permutation=torch.randperm(19)
    direct=model(context,state,times)
    permuted=model(context[permutation],state[permutation],times[permutation])
    assert torch.isfinite(direct).all()
    assert torch.allclose(permuted,direct[permutation],atol=2e-5,rtol=2e-5)


def test_flow_sdpa_backend_can_be_changed_without_parameter_changes():
    model=FlowMatchingTransformer(width=16,heads=4,depth=4)
    parameter_names=[name for name,_ in model.named_parameters()]
    model.set_sdpa_backend("math")
    blocks=(*model.encoder,model.middle,*model.decoder)
    assert all(block.sdpa_backend=="math" for block in blocks)
    assert [name for name,_ in model.named_parameters()]==parameter_names
    with pytest.raises(ValueError,match="unknown flow SDPA backend"):
        model.set_sdpa_backend("missing")


def test_optional_flow_compile_keeps_checkpoint_namespace(monkeypatch):
    model = BrainWMFoundationMoE(
        vocab_size=11, d_model=16, d_hidden=16, n_blocks=1,
        n_experts=2, attention_heads=4, flow_depth=2,
    )
    names = [name for name, _ in model.named_parameters()]
    calls = {}

    def fake_compile(function, **kwargs):
        calls.update(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model.set_flow_compile(True, mode="reduce-overhead", dynamic=True)
    assert model.flow_compile_enabled
    assert model.__dict__["_compiled_flow_forward"] is not None
    assert calls == {
        "mode": "reduce-overhead", "dynamic": True, "fullgraph": False,
    }
    assert [name for name, _ in model.named_parameters()] == names


def test_sinkhorn_runtime_disabled_matches_reference_and_is_not_a_module_child():
    torch.manual_seed(13)
    source = torch.randn(7, 16)
    target = torch.randn(7, 16)
    runtime = SinkhornRuntime(enabled=False, iterations=5)
    actual = runtime.plan(source, target)
    expected = sinkhorn_transport_plan(source, target, iterations=5)
    torch.testing.assert_close(actual, expected)
    assert runtime.status()["cached_graphs"] == 0


def test_sinkhorn_runtime_rejects_invalid_bucket_configuration():
    with pytest.raises(ValueError, match="compile buckets"):
        SinkhornRuntime(buckets=(0, 8))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is required")
def test_flash_and_cudnn_flow_sdpa_match_forward_and_backward():
    torch.manual_seed(11)
    cudnn=FlowMatchingTransformer(
        width=32,heads=4,depth=2,attention_mode="sdpa",sdpa_backend="cudnn",
    ).cuda()
    with torch.no_grad():
        for block in (*cudnn.encoder,cudnn.middle,*cudnn.decoder):
            block.modulation[-1].weight.normal_(std=.02)
            block.modulation[-1].bias.normal_(std=.02)
    flash=copy.deepcopy(cudnn)
    flash.set_sdpa_backend("flash")
    cudnn_inputs=[torch.randn(13,32,device="cuda",requires_grad=True)
                  for _ in range(2)]
    cudnn_inputs.append(torch.randn(13,3,device="cuda",requires_grad=True))
    flash_inputs=[value.detach().clone().requires_grad_(True)
                  for value in cudnn_inputs]

    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        cudnn_output=cudnn(*cudnn_inputs)
        flash_output=flash(*flash_inputs)
    torch.testing.assert_close(cudnn_output,flash_output,atol=2e-2,rtol=5e-2)
    cudnn_output.float().square().sum().backward()
    flash_output.float().square().sum().backward()
    for cudnn_input,flash_input in zip(cudnn_inputs,flash_inputs,strict=True):
        torch.testing.assert_close(
            cudnn_input.grad,flash_input.grad,atol=3e-2,rtol=8e-2,
        )
