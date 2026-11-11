import torch

import brainwm.router_acceleration as router
from brainwm.router_acceleration import CompiledTopKGate


def test_router_projection_logits_matches_reference_linear():
    torch.manual_seed(5)
    values = torch.randn(13, 8, dtype=torch.bfloat16)
    weights = torch.randn(4, 8)
    actual = router.router_projection_logits(values, weights)
    expected = torch.nn.functional.linear(values.float(), weights.float())
    torch.testing.assert_close(actual, expected)


def test_router_compile_policy_falls_back_without_changing_equation(monkeypatch):
    calls = {}

    def fake_compile(backend, function, **kwargs):
        calls["backend"] = backend
        calls["kwargs"] = kwargs
        return function

    monkeypatch.setattr(router, "compile_callable", fake_compile)
    values = torch.randn(32, 8)
    weights = torch.randn(4, 8)
    eager, eager_backend = router.run_router_projection_logits(
        values, weights, execution="eager",
    )
    compiled, compiled_backend = router.run_router_projection_logits(
        values, weights, execution="compile", mode="default",
    )
    assert eager_backend == "eager"
    assert compiled_backend == "torch_compile"
    assert calls["backend"] == "torch_compile"
    assert calls["kwargs"]["fullgraph"] is True
    torch.testing.assert_close(compiled, eager)


def test_router_auto_does_not_compile_cpu_inputs(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise AssertionError("CPU auto policy must stay eager")

    monkeypatch.setattr(router, "compile_callable", fail_compile)
    values = torch.randn(32, 8)
    weights = torch.randn(4, 8)
    _, backend = router.run_router_projection_logits(
        values, weights, execution="auto", min_tokens=1,
    )
    assert backend == "eager"


def test_compiled_gate_preserves_state_namespace_and_native_routing():
    from deepspeed.moe.sharded_moe import TopKGate

    torch.manual_seed(11)
    reference = TopKGate(
        8, 4, k=1, min_capacity=0, drop_tokens=False, use_rts=False,
    )
    candidate = CompiledTopKGate(reference, execution="eager")
    assert list(candidate.state_dict()) == list(reference.state_dict())
    candidate.load_state_dict(reference.state_dict())
    values = torch.randn(24, 8)
    reference_output = reference(values)
    candidate_output = candidate(values)
    for actual, expected in zip(candidate_output, reference_output, strict=True):
        torch.testing.assert_close(actual, expected)


def test_cuda_graph_rejects_cpu_inputs():
    values = torch.randn(4, 8)
    weights = torch.randn(3, 8)
    try:
        router.FixedShapeCudaGraph.capture(
            router.router_projection_logits, (values, weights),
        )
    except (RuntimeError, ValueError) as exc:
        assert "CUDA" in str(exc)
    else:  # pragma: no cover - only reached on an unusual CPU shim
        raise AssertionError("CPU CUDA-graph capture must fail")
