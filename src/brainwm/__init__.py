"""Brain World Model public API."""

from .annotation import (
    CrossModalAnnotationAdapter,
    HierarchicalResidualAnnotationHead,
    hierarchical_annotation_loss,
)
from .config import (
    AnnotationConfig,
    ExperimentConfig,
    ModalityAdapterConfig,
    ScFiTConfig,
    ScFiTTemporalConfig,
    ScUniFlowConfig,
    ScUniFlowGeneConfig,
    ScUniFlowSpatialConfig,
    load_config,
)
from .endpoint import (
    LatentEndpointAdapter,
    endpoint_velocity_from_endpoint,
    latent_endpoint_loss,
)
from .feature_completion import OrderedFeatureCompletionAdapter, masked_huber_loss
from .integration import (
    ScFiTTemporalAdapter,
    ScFiTTemporalOutput,
    ScUniFlowSparseGeneAdapter,
    ScUniFlowSpatialAdapter,
)
from .mamba_feature_completion import MambaFeatureCompletionAdapter
from .model import BrainWorldModel
from .modalities import NamedModalityRouter
from .ordered_mixer import OrderedGeneMixer
from .stage_residual import (
    GatedStageResidual,
    MultiStageResidualStack,
    absolute_state_delta,
    representation_anchor_loss,
)
from .losses import (
    developmental_flow_matching_loss,
    developmental_flow_matching_pair,
    developmental_flow_matching_target,
    spatial_adapter_smoothness_loss,
)
from .types import (
    BrainWorldModelOutput,
    HierarchicalAnnotationOutput,
    SparseExpressionBatch,
    TemporalTransitionOutput,
    WorldModelBatch,
)
from .training import measured_zero_token_view, unmeasured_gene_token_view

__all__ = [
    "BrainWorldModel",
    "BrainWorldModelOutput",
    "HierarchicalAnnotationOutput",
    "AnnotationConfig",
    "ModalityAdapterConfig",
    "ScFiTConfig",
    "ScFiTTemporalConfig",
    "ScUniFlowConfig",
    "ScUniFlowGeneConfig",
    "ScUniFlowSpatialConfig",
    "CrossModalAnnotationAdapter",
    "ExperimentConfig",
    "GatedStageResidual",
    "HierarchicalResidualAnnotationHead",
    "LatentEndpointAdapter",
    "MambaFeatureCompletionAdapter",
    "MultiStageResidualStack",
    "NamedModalityRouter",
    "OrderedFeatureCompletionAdapter",
    "OrderedGeneMixer",
    "ScFiTTemporalAdapter",
    "ScFiTTemporalOutput",
    "ScUniFlowSparseGeneAdapter",
    "ScUniFlowSpatialAdapter",
    "SparseExpressionBatch",
    "TemporalTransitionOutput",
    "WorldModelBatch",
    "measured_zero_token_view",
    "unmeasured_gene_token_view",
    "absolute_state_delta",
    "endpoint_velocity_from_endpoint",
    "hierarchical_annotation_loss",
    "latent_endpoint_loss",
    "load_config",
    "masked_huber_loss",
    "representation_anchor_loss",
    "developmental_flow_matching_loss",
    "developmental_flow_matching_pair",
    "developmental_flow_matching_target",
    "spatial_adapter_smoothness_loss",
]
from .flow_transformer import AdaLNZeroFlowBlock, FlowMatchingTransformer
from .backends import (
    ATTENTION_BACKENDS,
    COMPILE_BACKENDS,
    FLOW_BACKENDS,
    MOE_BACKENDS,
    OPTIMIZER_BACKENDS,
    backend_catalog,
    backend_names,
    backend_spec,
    create_backend,
    load_backend_entry_points,
    load_backend_plugins,
    register_attention_backend,
    register_compile_backend,
    register_flow_backend,
    register_moe_backend,
    register_optimizer_backend,
)

__all__.extend([
    "ATTENTION_BACKENDS",
    "COMPILE_BACKENDS",
    "FLOW_BACKENDS",
    "MOE_BACKENDS",
    "OPTIMIZER_BACKENDS",
    "backend_catalog",
    "backend_names",
    "backend_spec",
    "create_backend",
    "load_backend_entry_points",
    "load_backend_plugins",
    "register_attention_backend",
    "register_compile_backend",
    "register_flow_backend",
    "register_moe_backend",
    "register_optimizer_backend",
])
