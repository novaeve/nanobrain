import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from brainwm.foundation_moe import DifferentialLinearAttention
from brainwm.numpy_shard_loader import NumpyShardDataset, bucketed_attention_token_cost
from brainwm.sparse_objective import mask_expression
from brainwm.sparse_tokens import (
    SparseTokenBatch, add_csr_rows, concat_detached_suffix, expand_csr_rows,
)


def test_vendored_flash_bla_kernel_matches_recorded_upstream_hash():
    source=(Path(__file__).parents[1]/"src/flash_bla/ops/linear_attn/varlen.py")
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest=="e3a565e23896194af06ae4cf875b91513922fa7aef4bb9ed452e533d0da2d766"


def test_flash_bla_autotune_results_are_persistent():
    from flash_bla.ops.linear_attn import _varlen

    for name in (
        "_varlen_fwd_state_kernel","_varlen_fwd_output_kernel",
        "_varlen_bwd_state_kernel","_varlen_bwd_input_kernel",
    ):
        assert getattr(_varlen,name).cache_results


def test_packed_linear_attention_matches_padded_forward_and_backward():
    torch.manual_seed(7)
    lengths = torch.tensor([3, 5, 2])
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    padded = torch.randn(3, 5, 8, requires_grad=True)
    token_mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
    packed = padded[token_mask].detach().clone().requires_grad_(True)
    attention = DifferentialLinearAttention(8, 12, n_heads=3)
    layout = attention.build_packed_layout(row_ptr, len(packed), group_cells=2)

    padded_output = attention(padded, token_mask)
    packed_output = attention.forward_packed(packed, layout)
    torch.testing.assert_close(padded_output[token_mask], packed_output)

    padded_gradient, = torch.autograd.grad(padded_output[token_mask].square().sum(), padded)
    packed_gradient, = torch.autograd.grad(packed_output.square().sum(), packed)
    torch.testing.assert_close(padded_gradient[token_mask], packed_gradient)


def test_adaptive_packed_layout_coalesces_sparse_groups_within_budget():
    lengths = torch.tensor([2, 3, 4, 5])
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))

    fixed = DifferentialLinearAttention.build_packed_layout(
        row_ptr, int(row_ptr[-1]), group_cells=2,
    )
    adaptive = DifferentialLinearAttention.build_packed_layout(
        row_ptr, int(row_ptr[-1]), group_cells=2, max_attention_tokens=20,
    )

    assert len(fixed.groups) == 2
    assert fixed.allocated_tokens == 16
    assert len(adaptive.groups) == 1
    assert adaptive.allocated_tokens == 20


def test_adaptive_packed_layout_retains_dense_partition_at_budget():
    lengths = torch.tensor([1, 2, 100, 101])
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    layout = DifferentialLinearAttention.build_packed_layout(
        row_ptr, int(row_ptr[-1]), group_cells=2, max_attention_tokens=206,
    )

    assert len(layout.groups) == 2
    assert layout.allocated_tokens == 206


def test_adaptive_packed_attention_matches_fixed_forward_and_backward():
    torch.manual_seed(11)
    lengths = torch.tensor([2, 3, 4, 5])
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    fixed_input = torch.randn(int(row_ptr[-1]), 8, requires_grad=True)
    adaptive_input = fixed_input.detach().clone().requires_grad_(True)
    attention = DifferentialLinearAttention(8, 12, n_heads=3)
    fixed = attention.build_packed_layout(
        row_ptr, len(fixed_input), group_cells=2,
    )
    adaptive = attention.build_packed_layout(
        row_ptr, len(adaptive_input), group_cells=2, max_attention_tokens=20,
    )

    fixed_output = attention.forward_packed(fixed_input, fixed)
    adaptive_output = attention.forward_packed(adaptive_input, adaptive)
    torch.testing.assert_close(fixed_output, adaptive_output)

    fixed_gradient, = torch.autograd.grad(fixed_output.square().sum(), fixed_input)
    adaptive_gradient, = torch.autograd.grad(
        adaptive_output.square().sum(), adaptive_input,
    )
    torch.testing.assert_close(fixed_gradient, adaptive_gradient)


def test_packed_linear_attention_keeps_csr_segments_isolated():
    torch.manual_seed(13)
    row_ptr = torch.tensor([0, 2, 6])
    tokens = torch.randn(6, 8)
    changed = tokens.clone()
    changed[2:] += 100
    attention = DifferentialLinearAttention(8, 8, n_heads=2).eval()
    layout = attention.build_packed_layout(row_ptr, len(tokens), group_cells=1)

    original = attention.forward_packed(tokens, layout)
    modified = attention.forward_packed(changed, layout)
    torch.testing.assert_close(original[:2], modified[:2])


def test_concat_detached_suffix_has_prefix_only_backward():
    prefix = torch.randn(3, 5, requires_grad=True)
    suffix = torch.randn(2, 5, requires_grad=True)
    detached_suffix = suffix.detach()
    joined = concat_detached_suffix(prefix, detached_suffix)
    torch.testing.assert_close(joined, torch.cat((prefix, detached_suffix), dim=0))
    joined.square().sum().backward()
    torch.testing.assert_close(prefix.grad, 2 * prefix)
    assert suffix.grad is None


def test_concat_detached_suffix_rejects_attached_suffix():
    prefix = torch.randn(3, 5, requires_grad=True)
    suffix = torch.randn(2, 5, requires_grad=True)
    with pytest.raises(ValueError, match="detached suffix"):
        concat_detached_suffix(prefix, suffix)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is required")
def test_flash_bla_varlen_attention_matches_bucket_forward_and_backward():
    torch.manual_seed(19)
    lengths=torch.tensor([3,5,2],device="cuda",dtype=torch.long)
    row_ptr=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)))
    token_mask=torch.arange(5,device="cuda").unsqueeze(0)<lengths.unsqueeze(1)
    padded=torch.randn(3,5,8,device="cuda",requires_grad=True)
    packed=padded[token_mask].detach().clone().requires_grad_(True)
    row_ids=torch.repeat_interleave(torch.arange(3,device="cuda"),lengths)
    attention=DifferentialLinearAttention(8,12,n_heads=3).cuda()

    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        padded_output=attention(padded,token_mask)
        varlen_output=attention.forward_varlen(
            packed,row_ptr.to(torch.int32),lengths[row_ids],int(lengths.max()),
        )
    torch.testing.assert_close(
        padded_output[token_mask],varlen_output,atol=3e-2,rtol=8e-2,
    )

    padded_gradient,=torch.autograd.grad(
        padded_output[token_mask].float().square().sum(),padded,
    )
    packed_gradient,=torch.autograd.grad(
        varlen_output.float().square().sum(),packed,
    )
    torch.testing.assert_close(
        padded_gradient[token_mask],packed_gradient,atol=3e-2,rtol=8e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is required")
def test_flash_bla_varlen_attention_keeps_csr_segments_isolated():
    torch.manual_seed(23)
    row_ptr=torch.tensor([0,2,6],device="cuda",dtype=torch.int32)
    lengths=row_ptr[1:]-row_ptr[:-1]
    row_ids=torch.repeat_interleave(torch.arange(2,device="cuda"),lengths)
    tokens=torch.randn(6,8,device="cuda")
    changed=tokens.clone();changed[2:]+=100
    attention=DifferentialLinearAttention(8,8,n_heads=2).cuda().eval()

    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        original=attention.forward_varlen(
            tokens,row_ptr,lengths[row_ids],int(lengths.max()),
        )
        modified=attention.forward_varlen(
            changed,row_ptr,lengths[row_ids],int(lengths.max()),
        )
    torch.testing.assert_close(original[:2],modified[:2])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is required")
def test_flash_bla_reports_differential_head_grid_limit_before_launch():
    attention=DifferentialLinearAttention(8,8,n_heads=4).cuda()
    cu_seqlens=torch.zeros(8193,device="cuda",dtype=torch.int32)
    with pytest.raises(ValueError,match="at most 8191 cells"):
        attention.forward_varlen(
            torch.empty(0,8,device="cuda"),cu_seqlens,
            torch.empty(0,device="cuda",dtype=torch.long),0,
        )


def test_numpy_loader_packed_mode_uses_real_token_budget(tmp_path):
    np.save(tmp_path / "panel_source_id.npy", np.array([0]))
    np.save(tmp_path / "panel_ptr.npy", np.array([0, 8]))
    np.save(tmp_path / "panel_gene_index.npy", np.arange(8))
    stem = tmp_path / "train-shard-00000"
    np.save(f"{stem}-gene_index.npy", np.array([1, 2, 3, 4, 5, 6]))
    np.save(f"{stem}-values.npy", np.arange(1, 7, dtype=np.float32))
    np.save(f"{stem}-row_ptr.npy", np.array([0, 1, 5, 6]))
    np.save(f"{stem}-row_source_id.npy", np.zeros(3, dtype=np.int64))

    padded = next(iter(NumpyShardDataset(
        tmp_path, batch_size=3, zero_tokens_per_cell=0,
        max_padded_tokens=6, packed=False,
    )))
    packed = next(iter(NumpyShardDataset(
        tmp_path, batch_size=3, zero_tokens_per_cell=0,
        max_padded_tokens=6, packed=True, packed_group_cells=1,
    )))

    assert padded.batch_size == 1
    assert packed.batch_size == 3
    assert packed.packed
    assert packed.row_ptr.tolist() == [0, 1, 5, 6]
    assert packed.gene_index.tolist() == [1, 2, 3, 4, 5, 6]
    assert packed.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert packed.token_mask.all()
    assert packed.packed_layout_validated
    assert packed.packed_min_seqlen == 1
    assert packed.packed_max_seqlen == 4


def test_numpy_loader_bounds_packed_attention_slots(tmp_path):
    np.save(tmp_path / "panel_source_id.npy", np.array([0]))
    np.save(tmp_path / "panel_ptr.npy", np.array([0, 20]))
    np.save(tmp_path / "panel_gene_index.npy", np.arange(20))
    stem = tmp_path / "train-shard-00000"
    row_ptr = np.array([0, 1, 3, 7, 15])
    np.save(f"{stem}-gene_index.npy", np.arange(15))
    np.save(f"{stem}-values.npy", np.ones(15, dtype=np.float32))
    np.save(f"{stem}-row_ptr.npy", row_ptr)
    np.save(f"{stem}-row_source_id.npy", np.zeros(4, dtype=np.int64))

    batch = next(iter(NumpyShardDataset(
        tmp_path, batch_size=4, zero_tokens_per_cell=0,
        max_padded_tokens=12, packed=True, packed_group_cells=2,
    )))
    lengths = np.diff(batch.row_ptr.numpy())
    assert bucketed_attention_token_cost(lengths, 2) <= 12
    assert batch.batch_size == 3


def test_numpy_loader_finds_largest_prefix_when_bucket_cost_is_nonmonotonic(tmp_path):
    np.save(tmp_path / "panel_source_id.npy", np.array([0]))
    np.save(tmp_path / "panel_ptr.npy", np.array([0, 200]))
    np.save(tmp_path / "panel_gene_index.npy", np.arange(200))
    stem = tmp_path / "train-shard-00000"
    row_ptr = np.array([0, 1, 101, 103])
    np.save(f"{stem}-gene_index.npy", np.arange(103))
    np.save(f"{stem}-values.npy", np.ones(103, dtype=np.float32))
    np.save(f"{stem}-row_ptr.npy", row_ptr)
    np.save(f"{stem}-row_source_id.npy", np.zeros(3, dtype=np.int64))

    batch = next(iter(NumpyShardDataset(
        tmp_path, batch_size=3, zero_tokens_per_cell=0,
        max_padded_tokens=110, packed=True, packed_group_cells=2,
    )))
    assert bucketed_attention_token_cost([1, 100], 2) == 200
    assert bucketed_attention_token_cost([1, 100, 2], 2) == 104
    assert batch.batch_size == 3


def test_vectorized_masking_keeps_one_valid_token_per_nonempty_row():
    batch = SparseTokenBatch(
        gene_index=torch.arange(9), values=torch.ones(9),
        row_ptr=torch.tensor([0, 3, 6, 9]), measured_gene_index=torch.arange(9),
        token_mask=torch.tensor([
            True, False, False,
            True, True, False,
            False, False, False,
        ]),
    )
    masked = mask_expression(batch, probability=1e-12,
                             generator=torch.Generator().manual_seed(17))
    selected = masked.expression_masked.reshape(3, 3)
    assert selected.sum(1).tolist() == [1, 1, 0]
    assert not (masked.expression_masked & ~batch.token_mask).any()
    torch.testing.assert_close(
        masked.expression_mask_probability[batch.token_mask],
        torch.tensor([1.0, 0.5, 0.5]),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
))])
def test_csr_row_expansion_matches_advanced_index_forward_and_backward(device):
    row_ptr=torch.tensor([0,3,5,9],device=device)
    row_ids=torch.repeat_interleave(
        torch.arange(3,device=device),row_ptr[1:]-row_ptr[:-1],
    )
    reference=torch.randn(3,7,device=device,requires_grad=True)
    actual=reference.detach().clone().requires_grad_()
    reference_output=reference[row_ids]
    actual_output=expand_csr_rows(actual,row_ids,row_ptr)
    torch.testing.assert_close(actual_output,reference_output,atol=0,rtol=0)
    weights=torch.linspace(.1,1.0,len(row_ids),device=device).unsqueeze(1)
    (reference_output*weights).sum().backward()
    (actual_output*weights).sum().backward()
    torch.testing.assert_close(actual.grad,reference.grad,atol=0,rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
))])
def test_csr_addition_matches_expansion_and_keeps_compact_backward(device):
    torch.manual_seed(29)
    row_ptr=torch.tensor([0,3,5,9],device=device)
    row_ids=torch.repeat_interleave(
        torch.arange(3,device=device),row_ptr[1:]-row_ptr[:-1],
    )
    tokens=torch.randn(9,7,device=device,requires_grad=True)
    cells=torch.randn(3,7,device=device,requires_grad=True)
    expected=tokens+expand_csr_rows(cells,row_ids,row_ptr)
    actual=add_csr_rows(tokens,cells,row_ids,row_ptr)
    torch.testing.assert_close(actual,expected)
    expected_grads=torch.autograd.grad(
        expected.square().sum(),(tokens,cells),retain_graph=True,
    )
    actual_grads=torch.autograd.grad(actual.square().sum(),(tokens,cells))
    for actual_gradient,expected_gradient in zip(actual_grads,expected_grads,strict=True):
        torch.testing.assert_close(actual_gradient,expected_gradient)
    assert actual.grad_fn.__class__.__name__ == "_CSRAddExpandedBackward"
