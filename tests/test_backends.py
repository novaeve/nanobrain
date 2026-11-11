import torch

from brainwm.backends import (
    BackendRegistry,
    backend_catalog,
    backend_names,
    create_backend,
)
from brainwm.foundation_moe import BrainWMFoundationMoE
from brainwm.sparse_tokens import SparseTokenBatch


def test_registry_decorator_and_capability_manifest():
    registry = BackendRegistry("test")

    @registry.register(
        "reference", aliases=("ref",),
        capabilities={"backward": True}, source="test-suite",
    )
    def factory(value=0):
        return value + 1

    assert registry.create("ref", 4) == 5
    assert registry.resolve("reference").manifest() == {
        "kind": "test",
        "name": "reference",
        "aliases": ["ref"],
        "source": "test-suite",
        "capabilities": {"backward": True},
    }


def test_registry_filters_optional_common_keywords_for_small_factories():
    registry = BackendRegistry("filtered")
    registry.register("tiny", lambda width: width)
    assert registry.create(
        "tiny", width=9, unsupported_execution_policy="eager",
    ) == 9


def test_builtin_backend_catalog_has_reproducible_defaults():
    assert "flash_bla_differential" in backend_names("attention")
    assert "torch_sdpa" in backend_names("attention")
    assert "easy_deepspeed" in backend_names("moe")
    assert "fmt" in backend_names("flow")
    assert "adamw" in backend_names("optimizer")
    assert "torch_compile" in backend_names("compile")
    catalog = backend_catalog()
    assert all(kind in catalog for kind in ("attention", "moe", "flow", "optimizer", "compile"))
    assert any(item["name"] == "adamw" for item in catalog["optimizer"])


def test_pytorch_sdpa_backend_uses_bucket_fallback_for_packed_cpu_tokens():
    """A non-varlen attention plugin must not be routed into Flash-BLA."""
    model = BrainWMFoundationMoE(
        vocab_size=11,
        d_model=8,
        d_hidden=16,
        n_blocks=1,
        n_experts=2,
        top_k=1,
        attention_heads=2,
        attention_backend="torch_sdpa",
        packed_attention_backend="auto",
    ).eval()
    lengths = torch.tensor([2, 3], dtype=torch.long)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    tokens = int(row_ptr[-1])
    batch = SparseTokenBatch(
        gene_index=torch.arange(tokens) % 11,
        values=torch.linspace(0.1, 1.0, tokens),
        row_ptr=row_ptr,
        measured_gene_index=torch.arange(11),
        token_mask=torch.ones(tokens, dtype=torch.bool),
        packed=True,
        packed_layout_validated=True,
        packed_min_seqlen=int(lengths.min()),
        packed_max_seqlen=int(lengths.max()),
        row_ids=torch.repeat_interleave(torch.arange(2), lengths),
    )
    output = model.forward_sparse(batch, encode_only=True)
    assert output["attention_backend"] == "bucket"
    assert output["cell_embedding"].shape == (2, 8)
    assert model.backend_manifest()["attention"]["name"] == "torch_sdpa"


def test_simple_moe_block_forward_is_adapted_to_bucket_packing():
    """Blocks without a custom packed method still use the common protocol."""
    model = BrainWMFoundationMoE(
        vocab_size=11,
        d_model=8,
        d_hidden=16,
        n_blocks=1,
        n_experts=2,
        top_k=1,
        moe_backend="mage_beta",
        packed_attention_backend="auto",
    ).eval()
    lengths = torch.tensor([2, 3], dtype=torch.long)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    tokens = int(row_ptr[-1])
    batch = SparseTokenBatch(
        gene_index=torch.arange(tokens) % 11,
        values=torch.linspace(0.1, 1.0, tokens),
        row_ptr=row_ptr,
        measured_gene_index=torch.arange(11),
        token_mask=torch.ones(tokens, dtype=torch.bool),
        packed=True,
        packed_layout_validated=True,
        packed_min_seqlen=int(lengths.min()),
        packed_max_seqlen=int(lengths.max()),
        row_ids=torch.repeat_interleave(torch.arange(2), lengths),
    )
    output = model.forward_sparse(batch, encode_only=True)
    assert output["attention_backend"] == "bucket"
    assert output["cell_embedding"].shape == (2, 8)


def test_model_records_backend_choices_without_parameter_namespace_changes():
    torch.manual_seed(7)
    model = BrainWMFoundationMoE(
        vocab_size=11, d_model=16, d_hidden=16, n_blocks=1,
        n_experts=2, attention_heads=2,
    )
    names = tuple(name for name, _ in model.named_parameters())
    manifest = model.backend_manifest()
    assert manifest["attention"]["name"] == "flash_bla_differential"
    assert manifest["moe"]["name"] == "easy_deepspeed"
    assert manifest["flow"]["name"] == "fmt"
    assert model.model_config["compile_backend"] == "torch_compile"
    model.set_compile_backend("eager")
    assert model.model_config["compile_backend"] == "eager"
    assert tuple(name for name, _ in model.named_parameters()) == names


def test_registered_optimizer_factory_preserves_adamw_contract():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = create_backend("optimizer", "adamw", [parameter], lr=1e-3)
    assert isinstance(optimizer, torch.optim.AdamW)
