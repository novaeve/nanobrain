__all__ = [
    "load_protocol_bundle",
    "load_protocol_manifest",
    "RouteATrainer",
    "evaluate_artifact_dir",
    "save_prediction_artifact",
    "run_scnode_protocol",
    "run_prescient_protocol",
    "run_mioflow_protocol",
    "run_route_a_training",
]


def __getattr__(name):
    if name in {"load_protocol_bundle", "load_protocol_manifest"}:
        from .protocol import load_protocol_bundle, load_protocol_manifest

        return {
            "load_protocol_bundle": load_protocol_bundle,
            "load_protocol_manifest": load_protocol_manifest,
        }[name]

    if name == "RouteATrainer":
        from .train import RouteATrainer

        return RouteATrainer

    if name in {"evaluate_artifact_dir", "save_prediction_artifact"}:
        from .eval import evaluate_artifact_dir, save_prediction_artifact

        return {
            "evaluate_artifact_dir": evaluate_artifact_dir,
            "save_prediction_artifact": save_prediction_artifact,
        }[name]

    if name == "run_scnode_protocol":
        from .scnode import run_scnode_protocol

        return run_scnode_protocol

    if name == "run_prescient_protocol":
        from .prescient import run_prescient_protocol

        return run_prescient_protocol

    if name == "run_mioflow_protocol":
        from .mioflow import run_mioflow_protocol

        return run_mioflow_protocol

    if name == "run_route_a_training":
        from .main import run_route_a_training

        return run_route_a_training

    raise AttributeError(name)
