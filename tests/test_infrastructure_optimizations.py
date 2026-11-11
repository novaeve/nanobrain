import types

import pytest
import torch
from torch import nn

from brainwm.foundation_moe import (
    BatchedSwiGLUExperts,
    BrainWMFoundationMoE,
    ModalityAdapter,
    SwiGLU,
    _fused_nbeats_pair,
    feature_map_bucket_for_tokens,
    feature_map_bucket_padding_ratio,
)
from brainwm.sparse_tokens import SparseTokenBatch
from brainwm.optimizer_utils import (
    GradientHealthMonitor,
    gradient_health,
    resolve_gradient_health_execution,
)
from brainwm.spatial_attention import SpatialCellAttention
from brainwm.cuda_prefetch import CUDAPrefetcher, move_to_device, record_stream


def _tiny_sparse_batch() -> SparseTokenBatch:
    return SparseTokenBatch(
        gene_index=torch.tensor([0, 1, 2, 1, 2, 3]),
        values=torch.tensor([1.0, 0.5, 2.0, 0.25, 1.5, 0.75]),
        row_ptr=torch.tensor([0, 3, 6]),
        measured_gene_index=torch.arange(4),
        token_mask=torch.ones(6, dtype=torch.bool),
        packed=False,
        neighbor_edges_present=None,
    )


def test_spatial_empty_graph_is_exact_identity_without_projection():
    module = SpatialCellAttention(8)
    cells = torch.randn(4, 8)
    neighbors = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.zeros(4, 3, dtype=torch.bool)
    output = module(cells, neighbors, mask, edges_present=False)
    assert output is cells


def test_encode_only_matches_cell_embedding_and_skips_heads():
    torch.manual_seed(11)
    model = BrainWMFoundationMoE(
        7,
        d_model=4,
        d_hidden=8,
        n_blocks=1,
        n_experts=2,
        moe_backend="mage_beta",
        annotation_classes=3,
        anatomy_classes=2,
    ).eval()
    batch = _tiny_sparse_batch()
    full = model.forward_sparse(batch)
    encoded = model.forward_sparse(batch, encode_only=True)
    torch.testing.assert_close(encoded["cell_embedding"], full["cell_embedding"])
    assert "expression_reconstruction" not in encoded
    assert "canonical_annotation_logits" not in encoded
    assert "anatomy_logits" not in encoded
    assert encoded["attention_backend"] == "padded"
    assert encoded["attention_sequence_count"] == batch.batch_size
    assert encoded["attention_max_seqlen"] == 3
    assert encoded["attention_max_seqlen_bucket"] == 4


def test_padded_spatial_identity_uses_loader_edge_fact():
    torch.manual_seed(12)
    model = BrainWMFoundationMoE(
        7,
        d_model=4,
        d_hidden=8,
        n_blocks=1,
        n_experts=2,
        moe_backend="mage_beta",
    ).eval()
    batch = _tiny_sparse_batch()
    batch.neighbor_indices = torch.zeros(2, 1, dtype=torch.long)
    batch.neighbor_mask = torch.zeros(2, 1, dtype=torch.bool)
    batch.neighbor_indices_validated = True
    batch.neighbor_edges_present = False
    output = model.forward_sparse(batch, encode_only=True)
    torch.testing.assert_close(output["cell_embedding"], output["pre_spatial_embedding"])


def test_row_id_sidecar_matches_csr_expansion():
    batch = _tiny_sparse_batch()
    expected = torch.repeat_interleave(
        torch.arange(batch.batch_size),
        batch.row_ptr[1:] - batch.row_ptr[:-1],
    )
    batch.row_ids = expected.clone()
    torch.testing.assert_close(batch.token_row_ids(), expected)


def test_categorical_modalities_pack_matches_individual_adapters():
    torch.manual_seed(13)
    model = BrainWMFoundationMoE(
        7,
        d_model=4,
        d_hidden=8,
        n_blocks=1,
        n_experts=2,
        modality_dims={"assay": 3, "donor": 5, "continuous": 2},
        categorical_modalities={"assay", "donor"},
        moe_backend="mage_beta",
    )
    values = {
        "assay": torch.tensor([0, 2, 1]),
        "donor": torch.tensor([[4], [1], [3]]),
        "continuous": torch.randn(3, 2),
    }
    observed = {
        "assay": torch.tensor([True, True, False]),
        "donor": torch.tensor([True, False, True]),
        "continuous": torch.tensor([True, False, True]),
    }
    masked = {
        "assay": torch.tensor([False, True, False]),
        "donor": torch.tensor([False, False, True]),
        "continuous": torch.tensor([False, False, True]),
    }
    packed = model._modality_context(3, torch.device("cpu"), values, observed, masked)
    with torch.no_grad():
        reference = None
        for name, adapter in model.modalities.items():
            add = adapter(values[name], observed[name], masked[name])
            reference = add if reference is None else reference + add
    torch.testing.assert_close(packed, reference)
    assert model._cached_categorical_pack is None
    packed.square().mean().backward()
    assert model.modalities["assay"].category.weight.grad is not None
    assert model.modalities["donor"].category.weight.grad is not None


def test_prepared_grouped_cast_weights_are_reused_and_cleared():
    legacy = types.SimpleNamespace(
        deepspeed_experts=nn.ModuleList([SwiGLU(4, 2.0) for _ in range(2)]),
        num_local_experts=2,
        ep_size=1,
    )
    experts = BatchedSwiGLUExperts(legacy, "batched")
    experts.prepare_weights()
    first = experts.weights_for(torch.bfloat16)
    second = experts.weights_for(torch.bfloat16)
    assert first is second
    experts.clear_prepared_weights()
    assert experts._cached_cast_weights == {}


def test_grouped_experts_reject_expert_parallelism():
    legacy = types.SimpleNamespace(
        deepspeed_experts=nn.ModuleList([SwiGLU(4, 2.0)]),
        num_local_experts=1,
        ep_size=2,
    )
    with pytest.raises(ValueError, match="ep_size=1"):
        BatchedSwiGLUExperts(legacy, "batched")


def test_development_fusion_policies_are_configured_without_parameter_changes(tmp_path):
    torch.manual_seed(31)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=2,
        moe_backend="mage_beta", fuse_development_attention=True,
        fuse_development_target_attention=True,
    )
    keys = tuple(model.state_dict())
    model.set_fuse_development_attention(False)
    model.set_fuse_development_target_attention(False)
    model.set_feature_map_execution("auto")
    model.set_feature_map_min_tokens(17)
    model.set_feature_map_compile_mode("default")
    model.set_population_balance_mode("combined")
    assert model.model_config["fuse_development_attention"] is False
    assert model.model_config["fuse_development_target_attention"] is False
    assert model.model_config["feature_map_execution"] == "auto"
    assert model.model_config["feature_map_min_tokens"] == 17
    assert model.model_config["feature_map_compile_mode"] == "default"
    assert model.model_config["population_balance_mode"] == "combined"
    attention_blocks = [
        block.attention for block in model.blocks if hasattr(block, "attention")
    ]
    assert all(block.feature_map_compile_mode == "default" for block in attention_blocks)
    assert tuple(model.state_dict()) == keys

    path = tmp_path / "execution-policy.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, path)
    restored, payload = BrainWMFoundationMoE.from_checkpoint(path)
    assert payload["model_config"]["fuse_development_target_attention"] is False
    assert restored.fuse_development_attention is False
    assert restored.fuse_development_target_attention is False
    assert restored.feature_map_execution == "auto"
    assert restored.feature_map_min_tokens == 17
    assert restored.feature_map_compile_mode == "default"
    assert restored.population_balance_mode == "combined"
    assert tuple(restored.state_dict()) == keys


def test_feature_map_static_bucket_policy_is_bounded_and_checkpoint_safe(tmp_path):
    assert feature_map_bucket_for_tokens(131072, 65536) == 131072
    assert feature_map_bucket_for_tokens(644200, 65536) == 655360
    assert feature_map_bucket_padding_ratio(644200, 65536) < 1.02
    assert feature_map_bucket_padding_ratio(200000, 65536) > 1.10
    torch.manual_seed(35)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=2,
        feature_map_execution="auto", feature_map_min_tokens=17,
        feature_map_bucket_size=8, feature_map_max_padding_ratio=1.25,
        moe_backend="easy_deepspeed",
    )
    keys = tuple(model.state_dict())
    model.set_feature_map_bucket_size(16)
    model.set_feature_map_max_padding_ratio(1.10)
    model.set_differential_merge_execution("auto")
    model.set_differential_merge_min_tokens(4096)
    model.set_differential_merge_compile_mode("default")
    assert model.model_config["feature_map_bucket_size"] == 16
    assert model.model_config["feature_map_max_padding_ratio"] == 1.10
    assert model.blocks[0].attention.feature_map_bucket_size == 16
    assert model.blocks[0].attention.feature_map_max_padding_ratio == 1.10
    assert model.model_config["differential_merge_execution"] == "auto"
    assert model.model_config["differential_merge_min_tokens"] == 4096
    assert model.model_config["differential_merge_compile_mode"] == "default"
    assert model.blocks[0].attention.differential_merge_execution == "auto"
    path = tmp_path / "feature-map-policy.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, path)
    restored, payload = BrainWMFoundationMoE.from_checkpoint(path)
    assert payload["model_config"]["feature_map_bucket_size"] == 16
    assert restored.feature_map_bucket_size == 16
    assert restored.feature_map_max_padding_ratio == 1.10
    assert restored.differential_merge_execution == "auto"
    assert restored.differential_merge_min_tokens == 4096
    assert restored.differential_merge_compile_mode == "default"
    assert tuple(restored.state_dict()) == keys


def test_combined_population_balance_mode_propagates_without_parameter_changes():
    torch.manual_seed(33)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=2,
        moe_backend="easy_deepspeed", population_balance_mode="exact",
    )
    keys = tuple(model.state_dict())
    model.set_population_balance_mode("combined")
    assert model.model_config["population_balance_mode"] == "combined"
    assert model.blocks[0].population_balance_mode == "combined"
    assert tuple(model.state_dict()) == keys


def test_router_execution_policy_round_trips_without_checkpoint_parameters(tmp_path):
    torch.manual_seed(34)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=2,
        top_k=1, moe_backend="easy_deepspeed",
    )
    keys = tuple(model.state_dict())
    model.set_router_execution(
        "auto", min_tokens=32768, mode="default", dynamic=True,
    )
    assert model.router_runtime_status() == {
        "execution": "auto",
        "min_tokens": 32768,
        "compile_mode": "default",
        "dynamic": True,
        "backends": ["eager"],
    }
    path = tmp_path / "router-policy.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, path)
    restored, payload = BrainWMFoundationMoE.from_checkpoint(path)
    assert payload["model_config"]["router_execution"] == "auto"
    assert restored.router_runtime_status()["execution"] == "auto"
    assert restored.router_runtime_status()["min_tokens"] == 32768
    assert restored.router_runtime_status()["compile_mode"] == "default"
    assert restored.router_runtime_status()["dynamic"] is True
    assert tuple(restored.state_dict()) == keys


def test_population_balance_projection_is_equivalent_to_individual_projections():
    torch.manual_seed(32)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=3,
        top_k=1, moe_backend="easy_deepspeed",
    )
    block = model.blocks[0]
    populations = (torch.randn(2, 1, 4), torch.randn(3, 1, 4))
    expected = tuple(block._routing_auxiliary_loss(population) for population in populations)
    actual = block._routing_auxiliary_losses_many(populations)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


def test_population_balance_matches_dense_one_hot_equation():
    """The optimized routing path must preserve DeepSpeed's balance scalar."""
    torch.manual_seed(320)
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=3,
        top_k=1, moe_backend="easy_deepspeed",
    )
    block = model.blocks[0]
    cell_input = torch.randn(11, 1, 4)
    flat = cell_input.reshape(-1, 4)
    gate = block.moe.deepspeed_moe.gate
    probabilities = torch.nn.functional.linear(
        flat.float(), gate.wg.weight.float(), bias=None,
    ).softmax(dim=-1)
    first = probabilities.argmax(dim=-1)
    expert_count = probabilities.shape[-1]
    importance = probabilities.mean(dim=0)
    dense_load = torch.nn.functional.one_hot(
        first, num_classes=expert_count,
    ).to(probabilities.dtype).mean(dim=0)
    expected = (importance * dense_load).sum() * expert_count
    actual = block._routing_auxiliary_loss(cell_input)
    torch.testing.assert_close(actual, expected)


def test_observed_packed_modality_fast_path_matches_linear_projection():
    torch.manual_seed(37)
    adapter = ModalityAdapter(5)
    values = torch.randn(17, 1, requires_grad=True)
    output = adapter(values, observed=None, masked=None)
    reference = adapter.value(values.float())
    torch.testing.assert_close(output, reference)
    output.sum().backward()
    assert values.grad is not None


def test_cuda_prefetcher_is_exact_passthrough_on_cpu_and_moves_nested_values():
    batch = {"x": torch.tensor([1.0]), "meta": (torch.tensor([2]), "tag")}
    returned = list(CUDAPrefetcher([batch], "cpu", enabled=True))[0]
    torch.testing.assert_close(returned["x"], batch["x"])
    assert returned["meta"][1] == "tag"
    moved = move_to_device(batch, torch.device("cpu"), non_blocking=False)
    torch.testing.assert_close(moved["x"], batch["x"])
    torch.testing.assert_close(moved["meta"][0], batch["meta"][0])


def test_record_stream_is_noop_for_cpu_nested_batches():
    """The allocator lifetime hook must keep the CPU fallback side-effect free."""
    batch = {"x": torch.tensor([1.0]), "meta": (torch.tensor([2]), "tag")}
    # A CUDA stream is intentionally not constructed here; CPU tensors are
    # ignored before any stream method is reached.  This exercises the public
    # walker through the same nested container shapes used by prefetching.
    class DummyStream:
        pass
    record_stream(batch, DummyStream())


def test_gradient_health_samples_l2_but_checks_finite_on_every_path():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([1.0])
    norm, finite = gradient_health((first, second), compute_norm=True)
    assert bool(finite)
    torch.testing.assert_close(norm, torch.sqrt(torch.tensor(26.0)))
    cached = norm.detach()
    first.grad = torch.tensor([float("nan"), 0.0])
    reused, finite = gradient_health(
        (first, second), compute_norm=False, previous_norm=cached,
    )
    assert not bool(finite)
    torch.testing.assert_close(reused, cached)


def test_gradient_health_auto_falls_back_for_cpu_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], fused=False)
    assert resolve_gradient_health_execution("auto", optimizer) == "foreach"
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_gradient_health_execution("amp_foreach", optimizer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_amp_foreach_gradient_monitor_updates_and_skips_nonfinite():
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, fused=True)
    monitor = GradientHealthMonitor((parameter,), optimizer, execution="auto")
    assert monitor.execution == "amp_foreach"

    parameter.grad = torch.tensor([2.0], device="cuda")
    original_gradient = parameter.grad.clone()
    _, finite = monitor.check(compute_norm=False)
    finite_step = monitor.optimizer_step(
        optimizer, loss=parameter.new_tensor(1.0), finite_gradients=finite,
    )
    torch.cuda.synchronize()
    assert bool(finite_step)
    torch.testing.assert_close(parameter.grad, original_gradient)
    updated = parameter.detach().clone()
    assert not torch.equal(updated, torch.ones_like(updated))

    parameter.grad = torch.tensor([float("nan")], device="cuda")
    _, finite = monitor.check(compute_norm=False)
    finite_step = monitor.optimizer_step(
        optimizer, loss=parameter.new_tensor(1.0), finite_gradients=finite,
    )
    torch.cuda.synchronize()
    assert not bool(finite_step)
    torch.testing.assert_close(parameter, updated)

    parameter.grad = torch.tensor([2.0], device="cuda")
    _, finite = monitor.check(compute_norm=False)
    finite_step = monitor.optimizer_step(
        optimizer, loss=parameter.new_tensor(float("nan")),
        finite_gradients=finite,
    )
    torch.cuda.synchronize()
    assert not bool(finite_step)
    torch.testing.assert_close(parameter, updated)


def test_fused_nbeats_pair_matches_two_linear_heads_and_gradients():
    torch.manual_seed(41)
    backcast = nn.Linear(6, 6)
    forecast = nn.Linear(6, 6)
    features = torch.randn(3, 5, 6, requires_grad=True)
    reference_backcast = backcast(features)
    reference_forecast = forecast(features)
    fused_backcast, fused_forecast = _fused_nbeats_pair(
        backcast, forecast, features,
    )
    torch.testing.assert_close(fused_backcast, reference_backcast)
    torch.testing.assert_close(fused_forecast, reference_forecast)
    reference_loss = (reference_backcast.square() + reference_forecast.square()).sum()
    fused_loss = (fused_backcast.square() + fused_forecast.square()).sum()
    reference_grads = torch.autograd.grad(
        reference_loss, (features, backcast.weight, forecast.weight),
        retain_graph=True,
    )
    fused_grads = torch.autograd.grad(
        fused_loss, (features, backcast.weight, forecast.weight),
    )
    for actual, expected in zip(fused_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected)


def test_prepared_nbeats_heads_are_reused_and_cleared():
    torch.manual_seed(43)
    block = BrainWMFoundationMoE(
        7, d_model=6, d_hidden=8, n_blocks=1, n_experts=2,
        moe_backend="easy_deepspeed",
    ).blocks[0]
    features = torch.randn(3, 4, 6, requires_grad=True)
    uncached_backcast, uncached_forecast = block._nbeats_projection(features)
    block.prepare_nbeats_heads()
    prepared = block._cached_nbeats_projection
    cached_backcast, cached_forecast = block._nbeats_projection(features)
    assert block._cached_nbeats_projection is prepared
    torch.testing.assert_close(cached_backcast, uncached_backcast)
    torch.testing.assert_close(cached_forecast, uncached_forecast)
    block.clear_nbeats_heads()
    assert block._cached_nbeats_projection is None
