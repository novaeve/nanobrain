"""Model and training configuration."""

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    research_components: list = field(default_factory=list)
    research_plugins: dict = field(default_factory=dict)  # empty preserves the existing Cajal architecture
    n_genes: int = 65918
    n_batches: int = 57
    # Hierarchy sizes, coarse to fine; the last entry is the label vocabulary itself.
    cell_type_levels: list[int] = field(default_factory=lambda: [167])
    anatomy_levels: list[int] = field(default_factory=lambda: [91])

    d_model: int = 256
    n_heads: int = 4  # head dim 64: FA3 needs >= 64 and runs ~2.3x faster than 8 x 32 here
    n_prelude: int = 2
    n_core: int = 2
    n_coda: int = 2
    loops: int = 4  # loops used at evaluation; training samples from [min_loops, max_loops]
    min_loops: int = 2
    max_loops: int = 6

    n_experts: int = 32
    top_k: int = 4
    n_shared_experts: int = 1
    expert_hidden: int = 256
    router_bias_rate: float = 1e-3  # aux-loss-free balancing step (DeepSeek-V3)
    moe_backend: str = ""  # deepinfra grouped GEMM backend; "" picks one for the device
    router_score: str = "sqrtsoftplus"  # DeepSeek-V4; "sigmoid" is V3
    n_hash_layers: int = 1  # first prelude layers route by gene id (V4 hash routing)

    nbeats: bool = True  # backcast/forecast residual in every block
    loop_injection: bool = True  # re-inject the token embedding at each loop
    activation_checkpointing: bool = True  # recompute each block in backward to save memory
    recompute_scope: str = "core"  # "all" blocks, or only the looped "core" (its memory grows with loops)
    recompute_granularity: str = "moe"  # recompute whole "block", or only its "moe" branch (attention kept)
    selective_checkpointing: bool = False  # keep attention outputs instead of recomputing them
    compile_blocks: bool = True  # torch.compile each encoder block (fuses norms, SwiGLU, casts)
    attention_backend: str = "fa3"  # "flex" (BlockMask), "jagged" (SDPA varlen) or "fa3" (FA3 varlen, head dim >= 64)

    z_dim: int = 64
    decoder_hidden: int = 512
    # "separate": gene head on tokens + batch-conditioned decoder on z (Cajal v0). "unified": one decoder for
    # every expression output (masked genes, reconstruction, spatial), conditioned on batch and task, so no
    # output needs the encoder to carry batch (UnifiedExpressionDecoder).
    decoder: str = "separate"
    likelihood: str = "nb"  # "nb" on raw counts, or "mse_log" on log1p(cp10k)
    # Latent regularisation of the integration objective: "kl" to N(0, I) (scVI), "none" (deterministic mu,
    # no KL; pair with a train.objectives regulariser such as sigreg) or "mog" (KL to a learned mixture prior).
    latent_reg: str = "kl"
    mog_components: int = 32
    # > 0: hidden width of a batch discriminator on mu, behind gradient reversal (the batch_adversary
    # objective, DANN-style); 0 builds none.
    batch_adversary: int = 0
    residual_dim: int = 0  # > 0: a separate residual latent feeds the decoder (within-type variation)
    recon_detach_mu: bool = False  # reconstruction gradients stop at mu (only decoder and residual learn from it)

    spatial_d: int = 128
    spatial_layers: int = 2
    # > 0: the spatial loss also matches each hidden cell's predicted latent to its own latent mean
    # (stop-gradient target, JEPA-style), a denoised target next to its sparse counts; 0 keeps Cajal v0.
    spatial_latent: float = 0.0
    # True: the neighbourhood model predicts a correction to the mean of the visible neighbours' latents
    # (output layer zero-initialised, so it starts at that mean); False keeps Cajal v0's direct prediction.
    spatial_residual_mean: bool = False
    # Encoder input values: "log1p" (log1p CP10K, Cajal v0) or "pearson" (analytic Pearson residuals,
    # Lause et al. 2021 / sctransform: (x - n p_g) / sqrt(mu + mu^2 / 100), clipped to +-sqrt(N)); the gene
    # rates p_g and N come from the pretraining set and are kept in the checkpoint (encoder.gene_rate).
    input_norm: str = "log1p"


@dataclass
class DataConfig:
    root: str = "/data/brainwm/unified_1m"
    dissociated_cells: int = 128  # per step
    spatial_groups: int = 16  # per step; each group is a centre cell plus its neighbours
    neighbors: int = 16
    max_nonzero_tokens: int = 4096  # sampled when a cell has more measured non-zero genes
    zero_tokens: int = 64  # measured zeros sampled per cell
    gene_mask_ratio: float = 0.15  # legacy (pre-mask-plan configs); the plan in configs/masks sets ratios
    neighbor_mask_ratio: float = 0.3  # legacy, as above
    masks: str = "cajal_v0"  # mask plan: configs/masks/<name>.toml, or a .toml path
    gene_programs: str = ""  # gene -> program ids (.npy); default <root>/gene_programs.npy
    seed: int = 0
    cell_fraction: float = 1.0  # train on a fixed random fraction of the cells (data-scaling runs)


@dataclass
class TrainConfig:
    lr_schedule: str = "cosine"
    optimizer_params: dict = field(default_factory=dict)
    research_plugins: dict = field(default_factory=dict)  # declared tensor losses/optimizer, saved in checkpoint
    out_dir: str = "runs/debug"
    steps: int = 100_000
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    kl_weight: float = 1.0
    # > 0: the KL weight is a dual variable holding the KL near this many nats (beta-VAE capacity target,
    # GECO): it grows while the KL is above target and shrinks below; kl_weight is where it starts.
    kl_target: float = 0.0
    kl_dual_lr: float = 0.01
    kl_warmup_steps: int = 5000
    # Losses are combined as weight * scale * loss; the balancer (brainwm_unified.balance) sets the
    # scales. "initial" divides each task loss by its mean over the first `norm_steps` steps, then
    # freezes, so tasks start on the same scale with fixed weights.
    balancer: str = "initial"
    balancer_params: dict = field(default_factory=dict)
    task_weights: dict[str, float] = field(
        default_factory=lambda: {"gene": 1.0, "integration": 1.0, "cell_type": 1.0, "anatomy": 1.0, "spatial": 1.0}
    )
    # Curriculum: [{until = N, task_weights = {...}}, ...] in step order; before step N the phase's
    # weights replace task_weights. Empty keeps one set of weights for the whole run.
    task_schedule: list = field(default_factory=list)
    norm_steps: int = 200
    grad_clip: float = 1.0
    log_every: int = 10
    grad_probe_every: int = 500  # per-task gradient norms and cosines on shared parameters
    stop_at: int = 0  # > 0: stop after this many steps; the LR schedule still spans `steps`
    keep_every: int = 0  # keep a step<N>.pt every this many steps (0: every 10 x ckpt_every)
    keep_weights_only: bool = True  # kept step<N>.pt hold model + config only (~40%); latest.pt stays resumable
    # Extra objectives (brainwm_unified.regularizers), each {name, weight, **params}; added as
    # weight * scale * loss. With budget = {<loss>: <max>, ...} the weight is a dual variable
    # starting at `weight` (balance.DualWeight; optional dual_lr), shrinking when a budget is exceeded.
    # With start = N the objective is off (not computed) before step N.
    objectives: list = field(default_factory=list)
    ckpt_every: int = 1000
    num_workers: int = 8
    compile: bool = True
    optimizer: str = "grouped_muon"  # "muon" (torch, per matrix) or "grouped_muon" (batched Newton-Schulz)


def to_dict(*configs) -> dict:
    return {type(c).__name__: asdict(c) for c in configs}
