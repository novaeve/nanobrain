"""API-first baseline adapters for nanoBrain."""

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .annotation import CellTypistAdapter, XGBoostAdapter
from .foundation import GeneformerV110MAdapter, ScGPTBrainAdapter, ScGPTHumanAdapter, UCEAdapter
from .development import ScFiTAdapter, ScNODEAdapter, TrajectoryNetAdapter
from .integration import BBKNNAdapter, HarmonyAdapter, ScanoramaAdapter, ScviScanviAdapter
from .spatial import GraphSTAdapter, ScUniFlowAdapter, SpaGCNAdapter, STAGATEAdapter
from .registry import build, build_many, names, register

__all__ = [
    "AdapterContext",
    "Availability",
    "BaselineAdapter",
    "CellTypistAdapter",
    "GeneformerV110MAdapter",
    "GraphSTAdapter",
    "BBKNNAdapter",
    "HarmonyAdapter",
    "ScanoramaAdapter",
    "ScGPTBrainAdapter",
    "ScGPTHumanAdapter",
    "ScFiTAdapter",
    "ScNODEAdapter",
    "ScUniFlowAdapter",
    "ScviScanviAdapter",
    "TaskPrediction",
    "SpaGCNAdapter",
    "STAGATEAdapter",
    "TrajectoryNetAdapter",
    "UCEAdapter",
    "XGBoostAdapter",
    "build",
    "build_many",
    "names",
    "register",
]
