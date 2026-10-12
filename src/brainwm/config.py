"""Typed, externally editable configuration for the Brain World Model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Keep training artifacts on the NAS-backed project tree even when a caller
# starts a job from a different working directory.  Existing configuration
# files use both absolute paths and the historical relative ``runs/...``
# spelling, so normalization is deliberately limited to ``training.output_dir``.
BRAINWM_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BRAINWM_RUNS_ROOT = (BRAINWM_PACKAGE_ROOT / "runs").resolve()

SUPPORTED_CORPUS_SPECIES = frozenset({"human", "mouse", "macaque", "marmoset"})
SUPPORTED_CORPUS_READER_KINDS = frozenset(
    {"indexed_h5ad", "mouse_molecular", "mouse_spatial"}
)

# These transforms affect only the sparse encoder value features.  Raw counts
# remain the authoritative decoder target and library-size input for every
# route, so normalization experiments cannot silently change the task.
SUPPORTED_EXPRESSION_NORMALIZATIONS = frozenset(
    {"log1p_library", "log1p_cpm", "log1p_raw", "clr_nonzero"}
)


def resolve_training_output_dir(value: str | Path) -> Path:
    """Resolve a training output path under the project-local ``runs`` tree.

    Relative paths beginning with ``runs`` retain their historical meaning
    relative to ``brain_world_model`` (for example, ``runs/foo`` becomes
    ``<project>/brain_world_model/runs/foo``).  Other relative values are
    treated as run names under ``runs``.  Absolute paths remain supported, but
    all paths must resolve below the project-local runs directory.
    """
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        if raw.parts and raw.parts[0] == "runs":
            raw = BRAINWM_PACKAGE_ROOT / raw
        else:
            raw = BRAINWM_RUNS_ROOT / raw
    resolved = raw.resolve()
    try:
        resolved.relative_to(BRAINWM_RUNS_ROOT)
    except ValueError as exc:
        raise ValueError(
            "training.output_dir must resolve inside the project runs directory "
            f"{BRAINWM_RUNS_ROOT}; got {value!s} -> {resolved}"
        ) from exc
    return resolved


@dataclass
class DataConfig:
    # ``artifact_dir`` remains the legacy single-corpus entry point.  It is
    # optional only when ``corpora`` supplies one or more corpus specs.
    artifact_dir: str = ""
    split: str = "train"
    # Optional, append-only cohort manifest containing source-local cell keys
    # (source_id, row_index, sample_key, split).  Empty preserves the full
    # indexed training corpus and all historical schedules.
    cell_selection_path: str = ""
    assay_types: list[str] = field(default_factory=list)
    source_local_batches: bool = True
    batch_size: int = 16
    vocabulary_path: str = ""
    drop_last: bool = True
    shuffle_batches: bool = True
    balance_ddp_by_source_cost: bool = True
    assay_round_robin: bool = False
    # Optional exposure control for the complete multimodal corpus.  The
    # historical ``proportional`` mode follows cell counts; ``equal`` builds a
    # deterministic per-epoch schedule with equal scRNA/snRNA/spatialRNA
    # exposure (minority assays may be cycled, never imputed).
    assay_balance: str = "proportional"
    # Optional deterministic descriptor schedules.  Empty preserves the
    # historical cell-proportional shuffle; explicit grouping is useful for
    # multi-corpus ablations where a majority species must not dominate every
    # optimizer or validation window.
    batch_round_robin_by: str = ""
    validation_round_robin_by: str = ""
    spatial_order_path: str = ""
    optional_modality_path: str = ""
    # Optional source-defined gene set used only by marker-preserving
    # representation objectives.  It never changes the measured-gene mask.
    marker_gene_index_path: str = ""
    # YAML accepts either a list of mappings, a name->mapping dictionary, or a
    # compact name->path dictionary.  The original shape is retained in the
    # dataclass so resolved config manifests remain faithful to user input.
    corpora: list[dict[str, Any]] | dict[str, Any] | None = None
    # Optional append-only global gene vocabulary used by multi-species
    # readers.  Existence is checked by the reader that consumes it, keeping
    # config loading usable on submit hosts where the NAS is mounted later.
    unified_gene_vocabulary_path: str = ""
    # Empty means all species represented by the selected corpus/artifact.
    species_filter: list[str] = field(default_factory=list)
    # Optional projection of the indexed cell metadata.  Keeping this unset
    # preserves the historical full-cell table; mixed runs can list only the
    # columns needed by their objectives to avoid materializing large label
    # strings (the reader fills absent optional columns from source/sample
    # context).
    cell_columns: list[str] | None = None
    # Input value standardization used by the sparse encoder.  Raw counts are
    # always retained for the NB decoder/targets; this field changes only the
    # encoder view and is therefore a controlled Data-axis variable.
    normalization: str = "log1p_library"
    # Optional bounded reader over a frozen global benchmark row plan.  When
    # set, the training entry point uses the global benchmark split and never
    # reads the legacy ``cells.parquet.split`` column.  A positive per-split
    # cap is required so validation slices cannot accidentally materialize the
    # full source index.
    global_benchmark_dir: str = ""
    global_row_plan_dir: str = ""
    global_max_rows_per_split: int = 0
    global_task: str = ""
    global_row_group_batch_size: int = 1

    def validate(self) -> None:
        """Validate legacy and multi-corpus data settings.

        Paths are intentionally validated for shape rather than existence;
        artifacts may be mounted after a job is submitted.  Corpus entries,
        however, must carry enough explicit metadata for a reader to select a
        species-safe backend without guessing from arbitrary path names.
        """
        if self.artifact_dir is None:
            self.artifact_dir = ""
        elif not isinstance(self.artifact_dir, (str, Path)):
            raise TypeError("data.artifact_dir must be a path string")
        else:
            self.artifact_dir = str(self.artifact_dir).strip()

        if self.unified_gene_vocabulary_path is None:
            self.unified_gene_vocabulary_path = ""
        elif not isinstance(self.unified_gene_vocabulary_path, (str, Path)):
            raise TypeError(
                "data.unified_gene_vocabulary_path must be a path string"
            )
        else:
            self.unified_gene_vocabulary_path = str(
                self.unified_gene_vocabulary_path
            ).strip()

        for field_name in ("global_benchmark_dir", "global_row_plan_dir"):
            value = getattr(self, field_name)
            if value is None:
                value = ""
            if not isinstance(value, (str, Path)):
                raise TypeError(f"data.{field_name} must be a path string")
            setattr(self, field_name, str(value).strip())
        if bool(self.global_benchmark_dir) != bool(self.global_row_plan_dir):
            raise ValueError(
                "data.global_benchmark_dir and data.global_row_plan_dir "
                "must be provided together"
            )
        if self.global_benchmark_dir:
            if (
                isinstance(self.global_max_rows_per_split, bool)
                or not isinstance(self.global_max_rows_per_split, int)
                or self.global_max_rows_per_split < 1
            ):
                raise ValueError(
                    "data.global_max_rows_per_split must be a positive integer "
                    "for bounded global training"
                )
            if (
                isinstance(self.global_row_group_batch_size, bool)
                or not isinstance(self.global_row_group_batch_size, int)
                or self.global_row_group_batch_size < 1
            ):
                raise ValueError(
                    "data.global_row_group_batch_size must be a positive integer"
                )
            if self.split != "train":
                raise ValueError(
                    "global row-plan training requires data.split='train'"
                )
            if self.global_task is None:
                self.global_task = ""
            if not isinstance(self.global_task, str):
                raise TypeError("data.global_task must be a string")
            self.global_task = self.global_task.strip().lower()
            valid_tasks = {
                "", "integration", "annotation", "development", "spatial",
            }
            if self.global_task not in valid_tasks:
                raise ValueError(
                    "data.global_task must be empty or one of "
                    f"{sorted(valid_tasks - {''})}"
                )
        elif self.global_max_rows_per_split not in (0, None):
            raise ValueError(
                "data.global_max_rows_per_split requires global benchmark paths"
            )

        corpus_species = self._validate_corpora()
        if not self.artifact_dir and not corpus_species and not self.global_benchmark_dir:
            raise ValueError(
                "data.artifact_dir, a non-empty data.corpora, or a global benchmark "
                "reader is required"
            )
        if self.global_benchmark_dir and (self.artifact_dir or corpus_species):
            raise ValueError(
                "global benchmark reader cannot be combined with artifact_dir/corpora"
            )

        raw_filter = self.species_filter
        if raw_filter is None:
            raw_filter = []
        if isinstance(raw_filter, (str, bytes)) or not isinstance(
            raw_filter, (list, tuple)
        ):
            raise TypeError("data.species_filter must be a list of species names")
        normalized_filter: list[str] = []
        for value in raw_filter:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "data.species_filter entries must be non-empty strings"
                )
            species = value.strip().lower()
            if species not in SUPPORTED_CORPUS_SPECIES:
                raise ValueError(
                    f"unsupported species in data.species_filter: {species!r}"
                )
            if species in normalized_filter:
                raise ValueError(
                    f"duplicate species in data.species_filter: {species!r}"
                )
            normalized_filter.append(species)
        self.species_filter = normalized_filter
        if corpus_species and normalized_filter:
            unknown = set(normalized_filter) - corpus_species
            if unknown:
                raise ValueError(
                    "data.species_filter contains species absent from data.corpora: "
                    f"{sorted(unknown)}"
                )

        valid_round_robin_fields = {"", "assay_type", "species", "corpus"}
        for field_name in (
            "batch_round_robin_by", "validation_round_robin_by"
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                raw_value = ""
            if not isinstance(raw_value, str):
                raise TypeError(f"data.{field_name} must be a string")
            value = raw_value.strip()
            if value not in valid_round_robin_fields:
                raise ValueError(
                    f"data.{field_name} must be one of "
                    f"{sorted(valid_round_robin_fields)!r}"
                )
            setattr(self, field_name, value)
        if self.assay_round_robin:
            if self.batch_round_robin_by not in {"", "assay_type"}:
                raise ValueError(
                    "data.assay_round_robin conflicts with "
                    "data.batch_round_robin_by"
                )
            self.batch_round_robin_by = "assay_type"
        if not isinstance(self.assay_balance, str):
            raise TypeError("data.assay_balance must be a string")
        self.assay_balance = self.assay_balance.strip().lower()
        if self.assay_balance not in {"proportional", "equal"}:
            raise ValueError(
                "data.assay_balance must be proportional or equal"
            )
        if self.assay_balance == "equal" and self.batch_round_robin_by not in {
            "", "assay_type"
        }:
            raise ValueError(
                "data.assay_balance=equal conflicts with batch_round_robin_by="
                f"{self.batch_round_robin_by!r}"
            )

        raw_columns = self.cell_columns
        if raw_columns is not None:
            if isinstance(raw_columns, (str, bytes)) or not isinstance(
                raw_columns, (list, tuple)
            ):
                raise TypeError("data.cell_columns must be a list of column names")
            normalized_columns: list[str] = []
            for value in raw_columns:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "data.cell_columns entries must be non-empty strings"
                    )
                column = value.strip()
                if column in normalized_columns:
                    raise ValueError(
                        f"duplicate data.cell_columns entry: {column!r}"
                    )
                normalized_columns.append(column)
            required_columns = {"source_id", "row_index", "sample_key", "split"}
            missing_columns = sorted(required_columns - set(normalized_columns))
            if missing_columns:
                raise ValueError(
                    "data.cell_columns must include required columns: "
                    + ", ".join(missing_columns)
                )
            self.cell_columns = normalized_columns

    def _validate_corpora(self) -> set[str]:
        """Validate corpus shape and return normalized species names."""
        value = self.corpora
        if value is None:
            return set()
        if isinstance(value, (str, bytes)) or not isinstance(
            value, (Mapping, list, tuple)
        ):
            raise TypeError("data.corpora must be a YAML list or mapping")
        if not value:
            return set()

        entries: list[tuple[str | None, Mapping[str, Any]]] = []
        if isinstance(value, Mapping):
            # A mapping with a corpus field is a single spec; otherwise it is
            # the compact name->path/name->spec form.
            spec_keys = {"name", "artifact_dir", "path", "species", "reader_kind"}
            if any(key in value for key in spec_keys):
                entries.append((None, value))
            else:
                for name, payload in value.items():
                    if isinstance(payload, Mapping):
                        entries.append((str(name), payload))
                    elif isinstance(payload, (str, Path)):
                        entries.append(
                            (
                                str(name),
                                {"name": str(name), "artifact_dir": payload},
                            )
                        )
                    else:
                        raise TypeError(
                            "data.corpora mapping values must be paths or mappings"
                        )
        else:
            for position, payload in enumerate(value):
                if not isinstance(payload, Mapping):
                    raise TypeError(
                        "data.corpora list entries must be mappings "
                        f"(entry {position})"
                    )
                entries.append((None, payload))

        names: set[str] = set()
        species_values: set[str] = set()
        for key_name, payload in entries:
            name_value = payload.get("name", key_name)
            if not isinstance(name_value, str) or not name_value.strip():
                raise ValueError("each data.corpora entry needs a non-empty name")
            name = name_value.strip()
            if name in names:
                raise ValueError(f"duplicate data.corpora name: {name!r}")
            names.add(name)

            artifact = payload.get("artifact_dir", payload.get("path"))
            if not isinstance(artifact, (str, Path)) or not str(artifact).strip():
                raise ValueError(
                    f"data.corpora[{name!r}] needs a non-empty artifact_dir"
                )

            species_value = payload.get("species")
            if species_value is None:
                # Compact name->path dictionaries mirror the data adapter's
                # established human/mouse inference; arbitrary names must be
                # explicit to avoid silently assigning a species.
                lowered = name.lower()
                inferred = next(
                    (
                        candidate
                        for candidate in SUPPORTED_CORPUS_SPECIES
                        if lowered.startswith(candidate)
                    ),
                    None,
                )
                if inferred is None:
                    raise ValueError(
                        f"data.corpora[{name!r}] needs an explicit species"
                    )
                species = inferred
            elif isinstance(species_value, str) and species_value.strip():
                species = species_value.strip().lower()
            else:
                raise ValueError(
                    f"data.corpora[{name!r}].species must be a non-empty string"
                )
            if species not in SUPPORTED_CORPUS_SPECIES:
                raise ValueError(
                    f"unsupported species in data.corpora[{name!r}]: {species!r}"
                )
            species_values.add(species)

            reader_kind = payload.get("reader_kind", "indexed_h5ad")
            if not isinstance(reader_kind, str) or reader_kind not in SUPPORTED_CORPUS_READER_KINDS:
                raise ValueError(
                    f"unsupported reader_kind in data.corpora[{name!r}]: "
                    f"{reader_kind!r}"
                )
            offset = payload.get("source_id_offset")
            if offset is not None and (
                isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            ):
                raise ValueError(
                    f"data.corpora[{name!r}].source_id_offset must be a nonnegative integer"
                )
            prefix = payload.get("sample_key_prefix")
            if prefix is not None and (
                not isinstance(prefix, str) or not prefix.strip()
            ):
                raise ValueError(
                    f"data.corpora[{name!r}].sample_key_prefix must be non-empty"
                )
        return species_values


@dataclass
class EncoderConfig:
    gene_dim: int = 128
    value_dim: int = 32
    hidden_dim: int = 256
    latent_dim: int = 256
    token_layers: int = 2
    token_context_layers: int = 0
    optional_modality_dim: int = 0
    dropout: float = 0.1
    pooling: str = "attention"
    variational: bool = False
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    gene_features_enabled: bool = False
    gene_feature_dir: str = ""
    gene_type_enabled: bool = True
    homology_group_enabled: bool = False
    external_gene_embedding_path: str = ""
    # Optional continuous/rank sidecar.  It is deliberately absent from the
    # module graph when disabled so legacy checkpoints keep an identical
    # state-dict contract.
    rank_value_sidecar_enabled: bool = False
    rank_value_sidecar_hidden_dim: int = 0
    rank_value_sidecar_gate_init: float = 0.0
    latent_partitions: dict[str, int] = field(default_factory=lambda: {
        "tf": 48,
        "signal": 48,
        "cell_cycle": 32,
        "metabolic": 32,
        "residual": 96,
    })
    # The value transform is kept on the encoder so old checkpoints remain
    # state-dict compatible.  ExperimentConfig.validate synchronizes this with
    # DataConfig.normalization and rejects contradictory declarations.
    expression_normalization: str = "log1p_library"
    # Optional explicit missingness tokens.  Disabled by default so the
    # historical encoder graph/state-dict remains unchanged; when enabled,
    # ``mask_token`` (active MGM masking) and ``missing_token`` (unmeasured
    # assay genes) are learned independently.
    null_token_enabled: bool = False
    null_token_init_std: float = 0.02


@dataclass
class ModalityAdapterConfig:
    """Named dense-modality adapters for the shared BrainWM encoder.

    The sparse expression route remains unchanged.  Each entry in
    ``input_dims`` names one optional per-cell modality and declares its input
    width; the encoder projects every entry to the shared context width before
    fusing it into the latent state.  Keeping this branch opt-in preserves old
    checkpoint graphs and makes the data contract explicit.
    """

    enabled: bool = False
    input_dims: dict[str, int] = field(default_factory=dict)
    hidden_dim: int = 256
    dropout: float = 0.1
    gate_init: float = 0.0


@dataclass
class ContextConfig:
    categorical_vocab_sizes: dict[str, int] = field(default_factory=dict)
    categorical_embedding_dim: int = 32
    continuous_dim: int = 4
    output_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    latent_neutral_fields: list[str] = field(default_factory=list)
    latent_neutralize_continuous: bool = False


@dataclass
class GraphConfig:
    enabled: bool = True
    k_neighbors: int = 8
    hidden_dim: int = 256
    message_dim: int = 256
    ligand_receptor_dim: int = 0
    coordinate_dim: int = 2
    recompute_topology: bool = True
    distance_scale: float = 1.0


@dataclass
class HierarchyConfig:
    enabled: bool = True
    n_niches: int = 8
    hidden_dim: int = 256
    epsilon: float = 1.0e-8


@dataclass
class DynamicsConfig:
    enabled: bool = True
    mode: str = "ode"
    solver: str = "rk4"
    steps: int = 8
    hidden_dim: int = 512
    time_dim: int = 64
    diffusion_floor: float = 1.0e-4
    integration_time_scale_days: float = 365.2425


@dataclass
class FlowMatchingConfig:
    enabled: bool = False
    hidden_dim: int = 256
    time_dim: int = 32
    dropout: float = 0.1


@dataclass
class MaskedGeneModelingConfig:
    enabled: bool = False
    hidden_dim: int = 256
    negative_samples: int = 256
    mask_probability: float = 0.15
    max_tokens_per_batch: int = 2048


@dataclass
class AssayAlignmentConfig:
    enabled: bool = False
    assay_vocab_size: int = 0
    hidden_dim: int = 128
    gradient_reversal_scale: float = 1.0


@dataclass
class CrossSpeciesConfig:
    n_species: int = 1
    species_embedding_dim: int = 32
    adapter_rank: int = 16
    private_scale: float = 1.0


@dataclass
class InterventionConfig:
    control_dim: int = 0
    embedding_dim: int = 64
    use_delta_control: bool = True


@dataclass
class DecoderConfig:
    hidden_dim: int = 256
    gene_dim: int = 128
    min_dispersion: float = 1.0e-4
    tie_gene_embeddings: bool = True
    sample_distribution: bool = False
    n_lineages: int = 8
    # Optional controls for testing whether metadata context is bypassing the
    # molecular representation. Defaults preserve the historical decoder.
    context_scale: float = 1.0
    context_dropout: float = 0.0


@dataclass
class AnnotationConfig:
    """Optional hierarchical annotation readout for Stage-0 representations.

    The head is deliberately disabled by default.  When enabled, the three
    encoder depths (token-local, token-context, and fused latent) are routed to
    a residual hierarchy derived from the scBrain mechanism; this is a BrainWM
    annotation capability, not a replacement for the Stage-0/1/2 semantics.
    """

    enabled: bool = False
    # Taxonomy cardinalities are project-data dependent and therefore have no
    # fabricated default.  An enabled head must provide them explicitly.
    output_dims: list[int] = field(default_factory=list)
    hidden_dim: int = 256
    residual_mode: str = "feature"
    dropout: float = 0.0
    # Empty means unit weights in ``hierarchical_annotation_loss``.
    level_weights: list[float] = field(default_factory=list)


@dataclass
class ScFiTTemporalConfig:
    """Optional Stage-1 temporal capability derived from scFiT."""

    enabled: bool = False
    parameterization: str = "endpoint"
    # ``endpoint`` returns the predicted target directly; ``residual`` uses
    # a bounded gate and is useful for warm-starting from a Stage-1 checkpoint.
    mode: str = "residual"
    hidden_dim: int = 256
    time_dim: int = 32
    layers: int = 2
    dropout: float = 0.0
    flow_eps: float = 1.0e-3
    biological_time_scale: float = 365.2425
    gate_init: float = 0.0


@dataclass
class ScUniFlowGeneConfig:
    """Optional Stage-0 gene-token capability derived from scUniFlow."""

    enabled: bool = False
    order_mode: str = "biological"
    random_seed: int | None = None
    # A path to an ordered-position -> global-gene ``.npy`` permutation.  An
    # empty path falls back to ascending global gene IDs at runtime.
    gene_order_path: str = ""
    grid_shape: tuple[int, int] | list[int] | None = None
    ffn_multiplier: float = 1.0
    dropout: float = 0.0
    gate_init: float = 0.0
    # Optional bounded token budget for very long sparse rows.  Zero keeps the
    # historical full-row path; positive values retain only the earliest
    # tokens in the configured biological order for the adapter branch.
    max_tokens_per_row: int = 0


@dataclass
class ScUniFlowSpatialConfig:
    """Optional Stage-2 spatial-context capability derived from scUniFlow."""

    enabled: bool = False
    hidden_dim: int = 256
    dropout: float = 0.0
    gate_init: float = 0.0


# Short aliases make configuration files/readouts less verbose while keeping
# the explicit names used by the integration contract available.
ScFiTConfig = ScFiTTemporalConfig
ScUniFlowConfig = ScUniFlowSpatialConfig


@dataclass
class ModelConfig:
    n_genes: int
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    masked_gene_modeling: MaskedGeneModelingConfig = field(default_factory=MaskedGeneModelingConfig)
    assay_alignment: AssayAlignmentConfig = field(default_factory=AssayAlignmentConfig)
    cross_species: CrossSpeciesConfig = field(default_factory=CrossSpeciesConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    # Method integrations are absent from the default state-dict path.  The
    # long names are intentional: they make the stage and source method clear
    # in resolved YAML manifests.
    scbrain_depth_head: AnnotationConfig | None = None
    scfit_temporal_adapter: ScFiTTemporalConfig = field(
        default_factory=ScFiTTemporalConfig
    )
    scuniflow_gene_adapter: ScUniFlowGeneConfig = field(
        default_factory=ScUniFlowGeneConfig
    )
    scuniflow_spatial_adapter: ScUniFlowSpatialConfig = field(
        default_factory=ScUniFlowSpatialConfig
    )
    # Named dense-modality adapters are appended to preserve positional
    # compatibility for callers of the historical ModelConfig constructor.
    modality_adapters: ModalityAdapterConfig = field(
        default_factory=ModalityAdapterConfig
    )


@dataclass
class LossConfig:
    reconstruction: float = 1.0
    context_neutral_reconstruction: float = 0.0
    kl: float = 0.0
    contrastive: float = 0.0
    assay_adversarial: float = 0.0
    flow_matching: float = 0.0
    partition_independence: float = 0.0
    dynamics: float = 1.0
    multi_step: float = 0.5
    semigroup: float = 0.1
    spatial: float = 0.1
    population: float = 0.1
    alignment: float = 0.1
    cycle: float = 0.1
    intervention_effect: float = 0.1
    regulatory: float = 0.0
    masked_gene_modeling: float = 0.0
    latent_variance: float = 0.0
    latent_covariance: float = 0.0
    expression_geometry: float = 0.0
    marker_expression_geometry: float = 0.0
    # Optional method-integration objectives.  Zero keeps all historical
    # trainers and checkpoints unchanged; stage-specific runners may add the
    # corresponding term only when its supervision is available.
    hierarchical_annotation: float = 0.0
    temporal_endpoint: float = 0.0
    temporal_velocity: float = 0.0
    spatial_adapter: float = 0.0
    representation_anchor: float = 0.0
    # Optional direct supervision in the fixed source-7 expression panel.
    # This is distinct from latent ``dynamics``/``population`` terms so an
    # experiment can state explicitly when it optimizes the benchmark metric.
    expression_population: float = 0.0
    # Stage-1 developmental flow matching.  This is intentionally appended to
    # preserve positional compatibility with historical LossConfig callers;
    # it is separate from ``flow_matching`` (the Stage-0 latent-prior task).
    developmental_flow_matching: float = 0.0


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-2
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 500
    min_learning_rate_ratio: float = 0.1


@dataclass
class TrainingConfig:
    seed: int = 20260807
    epochs: int = 20
    max_steps: int = 0
    log_every: int = 20
    validate_every: int = 500
    early_stopping_patience: int = 0
    checkpoint_every: int = 1000
    validation_batches: int = 128
    # Checkpoint selection remains overall/cell-weighted by default.  Joint
    # multi-species experiments may opt into a macro average so the majority
    # corpus cannot decide every best checkpoint by itself.
    validation_selection_metric: str = "overall"
    # Disable this for benchmarks that report the same validation cohort and
    # therefore must use a predeclared final step instead of target-dependent
    # checkpoint selection.
    select_best_on_validation: bool = True
    gradient_accumulation_steps: int = 1
    masked_token_probability: float = 0.0
    # Compute latent spread penalties from an auxiliary dropout-disabled
    # encoder view.  Disabled by default to preserve historical runs.
    deterministic_spread_view: bool = False
    # Scale the first-view mask probability by per-cell library depth.  Zero
    # preserves the historical global mask probability exactly.
    depth_aware_masking_alpha: float = 0.0
    # Recalibrate depth-aware token probabilities so their token-weighted
    # batch mean equals masked_token_probability.  Disabled for checkpoint
    # and resolved-config compatibility with the original depth-aware branch.
    depth_aware_masking_preserve_budget: bool = False
    contrastive_temperature: float = 0.2
    # Contrastive view objective.  ``infonce`` preserves the historical
    # default; ``barlow_twins`` is an isolated, explicitly comparable
    # alternative for objective ablations.
    contrastive_objective: str = "infonce"
    barlow_redundancy_weight: float = 0.005
    contrastive_reference_batch_size: int = 0
    contrastive_panel_min_genes: int = 0
    contrastive_panel_max_genes: int = 0
    contrastive_neutralize_context: bool = False
    # Optional input corruption for training against assay-specific panels.
    # A positive probability samples from the source's measured genes rather
    # than the observed nonzero support, so zero/detection information can be
    # represented without treating unmeasured genes as zeros.
    panel_corruption_probability: float = 0.0
    panel_corruption_min_genes: int = 0
    panel_corruption_max_genes: int = 0
    panel_zero_token_budget: int = 0
    # Number of globally-unmeasured gene placeholders sampled per cell for a
    # corrupted input view.  Zero preserves the historical panel route.
    panel_missing_token_budget: int = 0
    kl_warmup_steps: int = 0
    train_components: list[str] = field(default_factory=lambda: ["encoder", "decoder"])
    precision: str = "bf16"
    resume_from: str = ""
    keep_last_checkpoints: int = 3
    output_dir: str = "runs/brainwm_v1"
    amp: bool = True
    device: str = "cuda"


@dataclass
class TemporalPairingConfig:
    plan_dir: str = ""
    initialize_from: str = ""
    allowed_time_statuses: list[str] = field(default_factory=lambda: [
        "exact", "approximate", "range_midpoint",
    ])
    min_cells_per_time: int = 128
    max_batches_per_interval: int = 16
    validation_max_batches_per_interval: int = 2
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 30
    # Frozen source-7 assets used only when ``losses.expression_population`` is
    # enabled.  Empty paths preserve all historical temporal configurations.
    expression_panel_path: str = ""
    expression_metric_assets_path: str = ""
    # Stage-1 can optionally evaluate the Stage-2 graph/niche hierarchy at
    # each temporal dynamics/flow-matching state.  ``never`` preserves the
    # historical molecular-only trainer; ``auto`` and ``always`` still fall
    # back to a strict no-op when coordinates are unavailable.
    spatial_structure_mode: str = "never"


@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    # Machine-readable scope metadata.  Reference methods inform an optional
    # BrainWM capability branch; they are never treated as paper reproductions.
    project_role: str = ""
    official_reproduction: bool = False
    losses: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    temporal_pairing: TemporalPairingConfig = field(default_factory=TemporalPairingConfig)

    def validate(self) -> None:
        # Validate after CLI overrides as well as after YAML construction.  A
        # canonical absolute value also makes resolved config manifests
        # independent of the caller's current working directory.
        if not isinstance(self.project_role, str):
            raise TypeError("project_role must be a string")
        if not isinstance(self.official_reproduction, bool):
            raise TypeError("official_reproduction must be a bool")
        if self.official_reproduction:
            raise ValueError(
                "BrainWM capability configs cannot enable official_reproduction"
            )
        self.data.validate()
        encoder = self.model.encoder
        # Keep the axis declared in the data section and the encoder section
        # synchronized.  A single canonical value is written to resolved
        # configs so two runs cannot claim different preprocessing while
        # carrying the same benchmark identity.
        data_normalization = str(self.data.normalization).strip().lower()
        encoder_normalization = str(encoder.expression_normalization).strip().lower()
        if data_normalization not in SUPPORTED_EXPRESSION_NORMALIZATIONS:
            raise ValueError(
                "data.normalization must be one of "
                f"{sorted(SUPPORTED_EXPRESSION_NORMALIZATIONS)}"
            )
        if encoder_normalization not in SUPPORTED_EXPRESSION_NORMALIZATIONS:
            raise ValueError(
                "model.encoder.expression_normalization must be one of "
                f"{sorted(SUPPORTED_EXPRESSION_NORMALIZATIONS)}"
            )
        # Legacy YAMLs have no encoder field and use the data default.  If one
        # side is explicitly changed, require the declaration to agree rather
        # than silently selecting whichever section happens to be read first.
        if data_normalization != encoder_normalization:
            data_is_default = data_normalization == "log1p_library"
            encoder_is_default = encoder_normalization == "log1p_library"
            if data_is_default and not encoder_is_default:
                self.data.normalization = encoder_normalization
                data_normalization = encoder_normalization
            elif encoder_is_default and not data_is_default:
                encoder_normalization = data_normalization
            else:
                raise ValueError(
                    "data.normalization and model.encoder.expression_normalization "
                    "must agree"
                )
        self.data.normalization = data_normalization
        encoder.expression_normalization = encoder_normalization
        self.training.output_dir = str(
            resolve_training_output_dir(self.training.output_dir)
        )
        if self.model.n_genes <= 0:
            raise ValueError("model.n_genes must be positive")
        if sum(encoder.latent_partitions.values()) != encoder.latent_dim:
            raise ValueError("encoder.latent_partitions must sum to encoder.latent_dim")
        if encoder.pooling not in {"attention", "mean"}:
            raise ValueError("encoder.pooling must be attention or mean")
        if encoder.token_context_layers < 0:
            raise ValueError("encoder.token_context_layers must be nonnegative")
        if encoder.rank_value_sidecar_hidden_dim < 0:
            raise ValueError(
                "encoder.rank_value_sidecar_hidden_dim must be nonnegative"
            )
        if not math.isfinite(encoder.rank_value_sidecar_gate_init):
            raise ValueError(
                "encoder.rank_value_sidecar_gate_init must be finite"
            )
        if (
            encoder.rank_value_sidecar_enabled
            and encoder.rank_value_sidecar_hidden_dim == 0
            and encoder.value_dim < 1
        ):
            raise ValueError(
                "encoder.value_dim must be positive for the rank/value sidecar"
            )
        if encoder.logvar_min >= encoder.logvar_max:
            raise ValueError("encoder.logvar_min must be below encoder.logvar_max")
        if not isinstance(encoder.null_token_enabled, bool):
            raise TypeError("encoder.null_token_enabled must be boolean")
        if not math.isfinite(encoder.null_token_init_std) or encoder.null_token_init_std < 0:
            raise ValueError(
                "encoder.null_token_init_std must be finite and nonnegative"
            )
        if encoder.optional_modality_dim and not self.data.optional_modality_path:
            raise ValueError(
                "data.optional_modality_path is required when optional_modality_dim is enabled"
            )
        if encoder.gene_features_enabled and not encoder.gene_feature_dir:
            raise ValueError(
                "encoder.gene_feature_dir is required when gene features are enabled"
            )
        if (
            encoder.gene_features_enabled
            and not encoder.gene_type_enabled
            and not encoder.homology_group_enabled
            and not encoder.external_gene_embedding_path
        ):
            raise ValueError(
                "enabled gene features require gene type, homology, or an external embedding"
            )
        modality_cfg = self.model.modality_adapters
        if not isinstance(modality_cfg, ModalityAdapterConfig):
            if isinstance(modality_cfg, dict):
                modality_cfg = ModalityAdapterConfig(**modality_cfg)
                self.model.modality_adapters = modality_cfg
            else:
                raise TypeError(
                    "model.modality_adapters must be ModalityAdapterConfig or mapping"
                )
        if modality_cfg.hidden_dim < 1:
            raise ValueError("model.modality_adapters.hidden_dim must be positive")
        if not 0.0 <= modality_cfg.dropout < 1.0:
            raise ValueError("model.modality_adapters.dropout must be in [0, 1)")
        if not math.isfinite(modality_cfg.gate_init):
            raise ValueError("model.modality_adapters.gate_init must be finite")
        if isinstance(modality_cfg.input_dims, (str, bytes)) or not isinstance(
            modality_cfg.input_dims, Mapping
        ):
            raise TypeError("model.modality_adapters.input_dims must be a mapping")
        normalized_modality_dims: dict[str, int] = {}
        for raw_name, raw_dim in modality_cfg.input_dims.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("modality adapter names must be non-empty")
            if name in normalized_modality_dims:
                raise ValueError(f"duplicate modality adapter name: {name!r}")
            if isinstance(raw_dim, bool) or int(raw_dim) != raw_dim or int(raw_dim) < 1:
                raise ValueError(
                    f"modality adapter dimension for {name!r} must be a positive integer"
                )
            normalized_modality_dims[name] = int(raw_dim)
        modality_cfg.input_dims = normalized_modality_dims
        if modality_cfg.enabled and not modality_cfg.input_dims:
            raise ValueError(
                "enabled modality adapters require at least one input dimension"
            )
        if not modality_cfg.enabled and modality_cfg.input_dims:
            raise ValueError(
                "modality_adapters.input_dims are set but modality_adapters.enabled is false"
            )
        unknown_neutral_fields = (
            set(self.model.context.latent_neutral_fields)
            - set(self.model.context.categorical_vocab_sizes)
        )
        if unknown_neutral_fields:
            raise ValueError(
                "context.latent_neutral_fields are not categorical fields: "
                f"{sorted(unknown_neutral_fields)}"
            )
        mgm = self.model.masked_gene_modeling
        if mgm.hidden_dim < 1 or mgm.negative_samples < 1 or mgm.max_tokens_per_batch < 1:
            raise ValueError("masked_gene_modeling dimensions must be positive")
        if not 0.0 < mgm.mask_probability < 1.0:
            raise ValueError("masked_gene_modeling.mask_probability must be in (0, 1)")
        dynamics = self.model.dynamics
        if dynamics.mode not in {"ode", "sde"}:
            raise ValueError("dynamics.mode must be ode or sde")
        if dynamics.solver not in {"euler", "rk4"}:
            raise ValueError("dynamics.solver must be euler or rk4")
        if dynamics.mode == "sde" and dynamics.solver != "euler":
            raise ValueError("SDE mode currently requires Euler-Maruyama (solver=euler)")
        if dynamics.steps < 1:
            raise ValueError("dynamics.steps must be >= 1")
        if dynamics.integration_time_scale_days <= 0:
            raise ValueError("dynamics.integration_time_scale_days must be positive")
        flow = self.model.flow_matching
        if flow.time_dim < 2 or flow.time_dim % 2:
            raise ValueError("flow_matching.time_dim must be a positive even integer")
        alignment = self.model.assay_alignment
        if alignment.enabled:
            expected_assays = self.model.context.categorical_vocab_sizes.get("assay_type")
            if not expected_assays or alignment.assay_vocab_size != expected_assays:
                raise ValueError(
                    "assay_alignment.assay_vocab_size must match the assay_type vocabulary"
                )
        if self.model.cross_species.n_species < 1:
            raise ValueError("cross_species.n_species must be >= 1")
        decoder = self.model.decoder
        if decoder.context_scale < 0.0:
            raise ValueError("decoder.context_scale must be nonnegative")
        if not 0.0 <= decoder.context_dropout < 1.0:
            raise ValueError("decoder.context_dropout must be in [0, 1)")
        # ``scbrain_depth_head`` is an explicit alias retained for integration
        # manifests.  Normalize it to the canonical annotation config before
        # validating dimensions; conflicting enabled definitions fail closed.
        depth_head = self.model.scbrain_depth_head
        if depth_head is not None and not isinstance(depth_head, AnnotationConfig):
            if isinstance(depth_head, dict):
                depth_head = AnnotationConfig(**depth_head)
                self.model.scbrain_depth_head = depth_head
            else:
                raise TypeError(
                    "model.scbrain_depth_head must be AnnotationConfig, mapping, or null"
                )
        annotation = self.model.annotation
        if not isinstance(annotation, AnnotationConfig):
            if isinstance(annotation, dict):
                annotation = AnnotationConfig(**annotation)
                self.model.annotation = annotation
            else:
                raise TypeError("model.annotation must be AnnotationConfig or mapping")
        if depth_head is not None:
            if annotation.enabled and annotation != depth_head:
                raise ValueError(
                    "model.annotation and model.scbrain_depth_head disagree"
                )
            if not annotation.enabled:
                annotation = depth_head
                self.model.annotation = annotation
        if not isinstance(annotation.enabled, bool):
            raise TypeError("model.annotation.enabled must be boolean")
        if annotation.hidden_dim < 1:
            raise ValueError("model.annotation.hidden_dim must be positive")
        if annotation.residual_mode not in {"none", "logit", "feature"}:
            raise ValueError(
                "model.annotation.residual_mode must be none, logit, or feature"
            )
        if not math.isfinite(float(annotation.dropout)) or not 0.0 <= float(
            annotation.dropout
        ) < 1.0:
            raise ValueError("model.annotation.dropout must be in [0, 1)")
        if isinstance(annotation.output_dims, (str, bytes)):
            raise TypeError("model.annotation.output_dims must be a sequence")
        try:
            annotation_dims = [int(value) for value in annotation.output_dims]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "model.annotation.output_dims must contain positive integers"
            ) from exc
        if any(value <= 0 for value in annotation_dims):
            raise ValueError(
                "model.annotation.output_dims must contain positive integers"
            )
        if annotation.enabled and not annotation_dims:
            raise ValueError(
                "model.annotation.output_dims is required when annotation is enabled"
            )
        if annotation.enabled and len(annotation_dims) > 3:
            raise ValueError(
                "model.annotation supports at most the three encoder depth states"
            )
        annotation.output_dims = annotation_dims
        if isinstance(annotation.level_weights, (str, bytes)):
            raise TypeError("model.annotation.level_weights must be a sequence")
        try:
            annotation_weights = [float(value) for value in annotation.level_weights]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "model.annotation.level_weights must contain finite nonnegative values"
            ) from exc
        if annotation_weights and len(annotation_weights) != len(annotation_dims):
            raise ValueError(
                "model.annotation.level_weights must match output_dims length"
            )
        if any(not math.isfinite(value) or value < 0 for value in annotation_weights):
            raise ValueError(
                "model.annotation.level_weights must contain finite nonnegative values"
            )
        annotation.level_weights = annotation_weights
        if self.losses.hierarchical_annotation and not annotation.enabled:
            raise ValueError(
                "losses.hierarchical_annotation requires model.annotation.enabled=true"
            )

        scfit = self.model.scfit_temporal_adapter
        if not isinstance(scfit, ScFiTTemporalConfig):
            if isinstance(scfit, dict):
                scfit = ScFiTTemporalConfig(**scfit)
                self.model.scfit_temporal_adapter = scfit
            else:
                raise TypeError("model.scfit_temporal_adapter must be a mapping")
        if not isinstance(scfit.enabled, bool):
            raise TypeError("model.scfit_temporal_adapter.enabled must be boolean")
        if scfit.parameterization not in {"endpoint", "direct_velocity"}:
            raise ValueError(
                "model.scfit_temporal_adapter.parameterization must be endpoint or direct_velocity"
            )
        if scfit.mode not in {"endpoint", "residual"}:
            raise ValueError(
                "model.scfit_temporal_adapter.mode must be endpoint or residual"
            )
        for name in ("hidden_dim", "time_dim", "layers"):
            value = getattr(scfit, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"model.scfit_temporal_adapter.{name} must be a positive integer"
                )
        for name in ("dropout", "flow_eps", "biological_time_scale", "gate_init"):
            value = float(getattr(scfit, name))
            if not math.isfinite(value):
                raise ValueError(
                    f"model.scfit_temporal_adapter.{name} must be finite"
                )
        if not 0.0 <= float(scfit.dropout) < 1.0:
            raise ValueError("model.scfit_temporal_adapter.dropout must be in [0, 1)")
        if not 0.0 < float(scfit.flow_eps) <= 1.0:
            raise ValueError(
                "model.scfit_temporal_adapter.flow_eps must be in (0, 1]"
            )
        if float(scfit.biological_time_scale) <= 0.0:
            raise ValueError(
                "model.scfit_temporal_adapter.biological_time_scale must be positive"
            )
        if not -1.0 < float(scfit.gate_init) < 1.0:
            raise ValueError(
                "model.scfit_temporal_adapter.gate_init must lie in (-1, 1)"
            )

        gene_adapter = self.model.scuniflow_gene_adapter
        if not isinstance(gene_adapter, ScUniFlowGeneConfig):
            if isinstance(gene_adapter, dict):
                gene_adapter = ScUniFlowGeneConfig(**gene_adapter)
                self.model.scuniflow_gene_adapter = gene_adapter
            else:
                raise TypeError("model.scuniflow_gene_adapter must be a mapping")
        if not isinstance(gene_adapter.enabled, bool):
            raise TypeError("model.scuniflow_gene_adapter.enabled must be boolean")
        if not isinstance(gene_adapter.order_mode, str):
            raise TypeError("model.scuniflow_gene_adapter.order_mode must be a string")
        if gene_adapter.order_mode.lower() not in {
            "biological", "bio", "random", "none", "identity", "no_order", "no-order"
        }:
            raise ValueError("unsupported model.scuniflow_gene_adapter.order_mode")
        if gene_adapter.random_seed is not None and (
            isinstance(gene_adapter.random_seed, bool)
            or not isinstance(gene_adapter.random_seed, int)
        ):
            raise ValueError("model.scuniflow_gene_adapter.random_seed must be an integer")
        if gene_adapter.order_mode.lower() == "random" and gene_adapter.random_seed is None:
            raise ValueError(
                "model.scuniflow_gene_adapter.random_seed is required for random order"
            )
        if gene_adapter.grid_shape is not None:
            shape = tuple(gene_adapter.grid_shape)
            if len(shape) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            ):
                raise ValueError(
                    "model.scuniflow_gene_adapter.grid_shape must contain two positive integers"
                )
            gene_adapter.grid_shape = shape
        if not math.isfinite(float(gene_adapter.ffn_multiplier)) or gene_adapter.ffn_multiplier < 0:
            raise ValueError("model.scuniflow_gene_adapter.ffn_multiplier must be nonnegative")
        if not 0.0 <= float(gene_adapter.dropout) < 1.0:
            raise ValueError("model.scuniflow_gene_adapter.dropout must be in [0, 1)")
        if not -1.0 < float(gene_adapter.gate_init) < 1.0:
            raise ValueError("model.scuniflow_gene_adapter.gate_init must lie in (-1, 1)")
        if (
            isinstance(gene_adapter.max_tokens_per_row, bool)
            or not isinstance(gene_adapter.max_tokens_per_row, int)
            or gene_adapter.max_tokens_per_row < 0
        ):
            raise ValueError(
                "model.scuniflow_gene_adapter.max_tokens_per_row must be a nonnegative integer"
            )
        if gene_adapter.gene_order_path is None:
            gene_adapter.gene_order_path = ""
        if not isinstance(gene_adapter.gene_order_path, (str, Path)):
            raise TypeError("model.scuniflow_gene_adapter.gene_order_path must be a path string")
        gene_adapter.gene_order_path = str(gene_adapter.gene_order_path).strip()

        spatial_adapter = self.model.scuniflow_spatial_adapter
        if not isinstance(spatial_adapter, ScUniFlowSpatialConfig):
            if isinstance(spatial_adapter, dict):
                spatial_adapter = ScUniFlowSpatialConfig(**spatial_adapter)
                self.model.scuniflow_spatial_adapter = spatial_adapter
            else:
                raise TypeError("model.scuniflow_spatial_adapter must be a mapping")
        if not isinstance(spatial_adapter.enabled, bool):
            raise TypeError("model.scuniflow_spatial_adapter.enabled must be boolean")
        if not isinstance(spatial_adapter.hidden_dim, int) or isinstance(
            spatial_adapter.hidden_dim, bool
        ) or spatial_adapter.hidden_dim < 1:
            raise ValueError("model.scuniflow_spatial_adapter.hidden_dim must be positive")
        if not 0.0 <= float(spatial_adapter.dropout) < 1.0:
            raise ValueError("model.scuniflow_spatial_adapter.dropout must be in [0, 1)")
        if not -1.0 < float(spatial_adapter.gate_init) < 1.0:
            raise ValueError("model.scuniflow_spatial_adapter.gate_init must lie in (-1, 1)")
        if (
            (self.losses.temporal_endpoint or self.losses.temporal_velocity)
            and not scfit.enabled
        ):
            raise ValueError(
                "temporal endpoint/velocity losses require "
                "model.scfit_temporal_adapter.enabled=true"
            )
        if self.losses.spatial_adapter and not spatial_adapter.enabled:
            raise ValueError(
                "losses.spatial_adapter requires "
                "model.scuniflow_spatial_adapter.enabled=true"
            )
        if self.data.batch_size < 1:
            raise ValueError("data.batch_size must be >= 1")
        if self.data.cell_selection_path and not Path(
            self.data.cell_selection_path
        ).is_file():
            raise FileNotFoundError(
                "data.cell_selection_path does not exist: "
                f"{self.data.cell_selection_path}"
            )
        if (
            not math.isfinite(self.losses.context_neutral_reconstruction)
            or self.losses.context_neutral_reconstruction < 0
        ):
            raise ValueError(
                "losses.context_neutral_reconstruction must be finite and nonnegative"
            )
        if self.losses.marker_expression_geometry < 0:
            raise ValueError("losses.marker_expression_geometry must be nonnegative")
        for name in (
            "hierarchical_annotation",
            "temporal_endpoint",
            "temporal_velocity",
            "developmental_flow_matching",
            "spatial_adapter",
            "representation_anchor",
            "expression_population",
        ):
            value = getattr(self.losses, name)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"losses.{name} must be finite and nonnegative")
        if self.losses.developmental_flow_matching and not dynamics.enabled:
            raise ValueError(
                "losses.developmental_flow_matching requires "
                "model.dynamics.enabled=true"
            )
        if self.losses.temporal_endpoint and not scfit.enabled:
            raise ValueError(
                "losses.temporal_endpoint requires "
                "model.scfit_temporal_adapter.enabled=true"
            )
        if self.losses.temporal_velocity and not scfit.enabled:
            raise ValueError(
                "losses.temporal_velocity requires "
                "model.scfit_temporal_adapter.enabled=true"
            )
        if self.losses.spatial_adapter and not spatial_adapter.enabled:
            raise ValueError(
                "losses.spatial_adapter requires "
                "model.scuniflow_spatial_adapter.enabled=true"
            )
        if self.losses.expression_population:
            panel_path = str(self.temporal_pairing.expression_panel_path).strip()
            metric_path = str(
                self.temporal_pairing.expression_metric_assets_path
            ).strip()
            if not panel_path or not metric_path:
                raise ValueError(
                    "temporal_pairing.expression_panel_path and "
                    "expression_metric_assets_path are required when "
                    "losses.expression_population is enabled"
                )
            if not Path(panel_path).is_file():
                raise FileNotFoundError(
                    "source-7 expression panel does not exist: " + panel_path
                )
            if not Path(metric_path).is_file():
                raise FileNotFoundError(
                    "source-7 metric assets do not exist: " + metric_path
                )
            self.temporal_pairing.expression_panel_path = panel_path
            self.temporal_pairing.expression_metric_assets_path = metric_path
        if self.losses.marker_expression_geometry and not self.data.marker_gene_index_path:
            raise ValueError(
                "data.marker_gene_index_path is required when marker_expression_geometry is enabled"
            )
        if (
            self.losses.marker_expression_geometry
            and not Path(self.data.marker_gene_index_path).is_file()
        ):
            raise FileNotFoundError(
                "marker gene set does not exist: "
                f"{self.data.marker_gene_index_path}"
            )
        if not self.data.source_local_batches:
            raise ValueError("indexed H5AD training requires data.source_local_batches=true")
        if self.optimizer.name.lower() != "adamw":
            raise ValueError("the implemented optimizer is adamw")
        if self.training.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be fp32, fp16, or bf16")
        if self.training.device not in {"cpu", "cuda"}:
            raise ValueError("training.device must be cpu or cuda")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.training.validation_batches < 0:
            raise ValueError("validation_batches must be >= 0")
        if self.training.validation_selection_metric not in {
            "overall", "macro_species"
        }:
            raise ValueError(
                "validation_selection_metric must be overall or macro_species"
            )
        if not isinstance(self.training.select_best_on_validation, bool):
            raise TypeError("select_best_on_validation must be boolean")
        if not 0.0 <= self.training.masked_token_probability < 1.0:
            raise ValueError("masked_token_probability must be in [0, 1)")
        if not isinstance(self.training.deterministic_spread_view, bool):
            raise TypeError("deterministic_spread_view must be boolean")
        if (
            not math.isfinite(self.training.depth_aware_masking_alpha)
            or self.training.depth_aware_masking_alpha < 0.0
        ):
            raise ValueError(
                "depth_aware_masking_alpha must be finite and nonnegative"
            )
        if not isinstance(
            self.training.depth_aware_masking_preserve_budget, bool
        ):
            raise TypeError(
                "depth_aware_masking_preserve_budget must be boolean"
            )
        if self.training.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        if self.training.contrastive_objective not in {"infonce", "barlow_twins"}:
            raise ValueError(
                "contrastive_objective must be infonce or barlow_twins"
            )
        if self.training.barlow_redundancy_weight < 0:
            raise ValueError("barlow_redundancy_weight must be nonnegative")
        if self.training.contrastive_reference_batch_size < 0:
            raise ValueError("contrastive_reference_batch_size must be nonnegative")
        panel_min = self.training.contrastive_panel_min_genes
        panel_max = self.training.contrastive_panel_max_genes
        if panel_min < 0 or panel_max < 0:
            raise ValueError("contrastive panel gene counts must be nonnegative")
        if (panel_min == 0) != (panel_max == 0):
            raise ValueError(
                "contrastive panel min/max genes must both be zero or both be positive"
            )
        if panel_max and panel_min > panel_max:
            raise ValueError(
                "contrastive_panel_min_genes cannot exceed the maximum"
            )
        corruption_probability = self.training.panel_corruption_probability
        if not 0.0 <= corruption_probability <= 1.0:
            raise ValueError("panel_corruption_probability must be in [0, 1]")
        corruption_min = self.training.panel_corruption_min_genes
        corruption_max = self.training.panel_corruption_max_genes
        if corruption_min < 0 or corruption_max < 0:
            raise ValueError("panel corruption gene counts must be nonnegative")
        if (corruption_min == 0) != (corruption_max == 0):
            raise ValueError(
                "panel corruption min/max genes must both be zero or both be positive"
            )
        if corruption_max and corruption_min > corruption_max:
            raise ValueError(
                "panel_corruption_min_genes cannot exceed the maximum"
            )
        if corruption_probability and not corruption_max:
            raise ValueError(
                "panel corruption requires positive min/max gene limits"
            )
        if self.training.panel_zero_token_budget < 0:
            raise ValueError("panel_zero_token_budget must be nonnegative")
        if self.training.panel_missing_token_budget < 0:
            raise ValueError("panel_missing_token_budget must be nonnegative")
        if (
            self.training.panel_missing_token_budget > 0
            and not self.model.encoder.null_token_enabled
        ):
            raise ValueError(
                "panel_missing_token_budget requires encoder.null_token_enabled"
            )
        if self.training.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be nonnegative")
        if self.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        temporal = self.temporal_pairing
        if temporal.min_cells_per_time < 1:
            raise ValueError("temporal_pairing.min_cells_per_time must be positive")
        if temporal.max_batches_per_interval < 1:
            raise ValueError("temporal_pairing.max_batches_per_interval must be positive")
        if temporal.validation_max_batches_per_interval < 1:
            raise ValueError(
                "temporal_pairing.validation_max_batches_per_interval must be positive"
            )
        if temporal.sinkhorn_epsilon <= 0:
            raise ValueError("temporal_pairing.sinkhorn_epsilon must be positive")
        if temporal.sinkhorn_iterations < 1:
            raise ValueError("temporal_pairing.sinkhorn_iterations must be positive")
        if temporal.spatial_structure_mode not in {"auto", "always", "never"}:
            raise ValueError(
                "temporal_pairing.spatial_structure_mode must be auto, always, or never"
            )
        valid_components = {
            "encoder", "decoder", "graph", "hierarchy", "dynamics",
            "flow_matching", "assay_discriminator", "lineage_head", "mgm_head",
            "annotation_head", "scbrain_depth_head", "scfit_temporal_adapter",
            "scuniflow_gene_adapter", "scuniflow_spatial_adapter",
        }
        unknown_components = set(self.training.train_components) - valid_components
        if unknown_components:
            raise ValueError(f"unknown training components: {sorted(unknown_components)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(raw: dict[str, Any]) -> ExperimentConfig:
    model_raw = dict(raw["model"])
    # Accept both the concise ``annotation`` spelling used by earlier
    # experiments and the explicit integration-contract aliases.  The aliases
    # are normalized to dataclass instances before validation so callers can
    # safely consume a resolved config regardless of the YAML spelling.
    annotation_raw = model_raw.get("annotation", {})
    if isinstance(annotation_raw, AnnotationConfig):
        annotation = annotation_raw
    elif isinstance(annotation_raw, dict):
        annotation = AnnotationConfig(**annotation_raw)
    else:
        raise TypeError("model.annotation must be a mapping")
    depth_head_raw = model_raw.get("scbrain_depth_head")
    if depth_head_raw is None:
        depth_head = None
    elif isinstance(depth_head_raw, AnnotationConfig):
        depth_head = depth_head_raw
    elif isinstance(depth_head_raw, dict):
        depth_head = AnnotationConfig(**depth_head_raw)
    else:
        raise TypeError("model.scbrain_depth_head must be a mapping or null")

    def _nested(raw_value: Any, cls: type[Any], name: str) -> Any:
        if isinstance(raw_value, cls):
            return raw_value
        if isinstance(raw_value, dict):
            return cls(**raw_value)
        raise TypeError(f"model.{name} must be a mapping")

    model = ModelConfig(
        n_genes=int(model_raw["n_genes"]),
        encoder=EncoderConfig(**model_raw.get("encoder", {})),
        modality_adapters=ModalityAdapterConfig(
            **model_raw.get("modality_adapters", {})
        ),
        context=ContextConfig(**model_raw.get("context", {})),
        graph=GraphConfig(**model_raw.get("graph", {})),
        hierarchy=HierarchyConfig(**model_raw.get("hierarchy", {})),
        dynamics=DynamicsConfig(**model_raw.get("dynamics", {})),
        flow_matching=FlowMatchingConfig(**model_raw.get("flow_matching", {})),
        masked_gene_modeling=MaskedGeneModelingConfig(
            **model_raw.get("masked_gene_modeling", {})
        ),
        assay_alignment=AssayAlignmentConfig(**model_raw.get("assay_alignment", {})),
        cross_species=CrossSpeciesConfig(**model_raw.get("cross_species", {})),
        intervention=InterventionConfig(**model_raw.get("intervention", {})),
        decoder=DecoderConfig(**model_raw.get("decoder", {})),
        annotation=annotation,
        scbrain_depth_head=depth_head,
        scfit_temporal_adapter=_nested(
            model_raw.get("scfit_temporal_adapter", {}),
            ScFiTTemporalConfig,
            "scfit_temporal_adapter",
        ),
        scuniflow_gene_adapter=_nested(
            model_raw.get("scuniflow_gene_adapter", {}),
            ScUniFlowGeneConfig,
            "scuniflow_gene_adapter",
        ),
        scuniflow_spatial_adapter=_nested(
            model_raw.get("scuniflow_spatial_adapter", {}),
            ScUniFlowSpatialConfig,
            "scuniflow_spatial_adapter",
        ),
    )
    config = ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        data=DataConfig(**raw["data"]),
        model=model,
        project_role=str(raw.get("project_role", "")),
        official_reproduction=bool(raw.get("official_reproduction", False)),
        losses=LossConfig(**raw.get("losses", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        training=TrainingConfig(**raw.get("training", {})),
        temporal_pairing=TemporalPairingConfig(**raw.get("temporal_pairing", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    return _construct(raw)
