"""The Cajal campaign definition reproduces the constants that used to live in code."""

from brainwm_unified.objectives import TASKS as SEED_TASKS
from nanobrain.agent import react
from nanobrain.infra import contracts as C

# Verbatim copy of the literals removed from nanobrain/infra/contracts.py in the 2b refactor.
TASKS = ("gene", "integration", "cell_type", "anatomy", "spatial")
METRICS = (
    "Cell type fine macro-F1",
    "Region fine macro-F1",
    "Integration bio",
    "Integration batch",
    "Gene imputation Pearson (cell median)",
    "Spatial imputation Pearson (cell median)",
)
MODEL_SPACE = {
    "d_model": [128, 256, 384, 512],
    "n_heads": [2, 4, 6, 8],
    "n_prelude": [1, 2, 3, 4],
    "n_core": [1, 2, 3, 4],
    "n_coda": [1, 2, 3, 4],
    "loops": [2, 3, 4, 5, 6],
    "min_loops": [1, 2, 3, 4],
    "max_loops": [2, 3, 4, 5, 6, 8],
    "n_experts": [8, 16, 32, 64],
    "top_k": [1, 2, 4, 8],
    "n_shared_experts": [1, 2],
    "expert_hidden": [128, 256, 384, 512],
    "router_score": ["sqrtsoftplus", "sigmoid"],
    "n_hash_layers": [0, 1, 2],
    "nbeats": [False, True],
    "loop_injection": [False, True],
    "z_dim": [32, 64, 128],
    "decoder_hidden": [256, 512, 768, 1024],
    "decoder": ["separate", "unified"],
    "latent_reg": ["kl", "mog", "none"],
    "mog_components": [8, 16, 32, 64],
    "batch_adversary": [0, 64, 128, 256],
    "residual_dim": [0, 16, 32, 64],
    "recon_detach_mu": [False, True],
    "spatial_d": [64, 128, 256],
    "spatial_layers": [1, 2, 3, 4],
    "spatial_residual_mean": [False, True],
    "input_norm": ["log1p", "pearson"],
}
MODEL_RANGES = {"router_bias_rate": (0.0, 0.01), "spatial_latent": (0.0, 5.0)}
TRAIN_RANGES = {
    "lr": (1e-6, 0.02),
    "weight_decay": (0.0, 0.3),
    "grad_clip": (0.1, 20.0),
    "kl_weight": (0.001, 20.0),
    "kl_target": (0.0, 300.0),
    "kl_dual_lr": (0.0001, 0.1),
    "min_lr_ratio": (0.01, 0.5),
}
TRAIN_INTS = {"warmup_steps": (0, 4000), "kl_warmup_steps": (100, 10000), "norm_steps": (20, 1000)}
DATA_RANGES = {
    "dissociated_cells": (16, 512),
    "spatial_groups": (2, 64),
    "max_nonzero_tokens": (512, 8192),
    "zero_tokens": (8, 256),
}
OPTIMIZER_RANGES = {
    "adam_beta1": (0.8, 0.99),
    "adam_beta2": (0.9, 0.9999),
    "adam_eps": (1e-9, 1e-6),
    "muon_momentum": (0.8, 0.99),
    "muon_ns_steps": (3, 8),
    "sgd_momentum": (0.0, 0.99),
}
BALANCER_RANGES = {
    "grad_equal": {"every": (1, 200), "beta": (0.5, 0.99)},
    "famo": {"w_lr": (0.001, 0.1), "gamma": (0.0, 0.1), "eps": (1e-10, 1e-5)},
    "initial": {},
}
KINDS = ("token_transform", "gene_loss", "cell_type_loss", "anatomy_loss", "optimizer")

GROUPS = {
    "annotation": METRICS[:2],
    "integration": METRICS[2:4],
    "genes": METRICS[4:5],
    "spatial": METRICS[5:],
}


def test_definition_reproduces_former_constants():
    assert C.TASKS == TASKS and C.METRICS == METRICS and C.KINDS == KINDS
    assert C.MODEL_SPACE == MODEL_SPACE and C.MODEL_RANGES == MODEL_RANGES
    assert C.TRAIN_RANGES == TRAIN_RANGES and C.TRAIN_INTS == TRAIN_INTS
    assert C.DATA_RANGES == DATA_RANGES and C.OPTIMIZER_RANGES == OPTIMIZER_RANGES
    assert C.BALANCER_RANGES == BALANCER_RANGES
    for mapping in (C.MODEL_RANGES, C.TRAIN_RANGES, C.TRAIN_INTS, C.DATA_RANGES, C.OPTIMIZER_RANGES):
        for value in mapping.values():
            assert type(value) is tuple
    for key, values in C.MODEL_SPACE.items():
        assert [type(v) for v in values] == [type(v) for v in MODEL_SPACE[key]]
    assert C.METRIC_GROUPS == GROUPS
    assert C.METRIC_RANGES == {k: (-1.0 if "Pearson" in k else 0.0, 1.0) for k in METRICS}
    assert C.GPU_WHITELIST == (0, 1, 2, 3, 4) and C.GATE_FLOOR == 0.005


def test_objectives_match_the_seed_model():
    assert C.TASKS == tuple(SEED_TASKS)


def test_agent_brief_is_the_former_system_prompt():
    assert react.SYSTEM.startswith("Evolve the existing Cajal v1 shared encoder")
    assert not react.SYSTEM.endswith("\n")
