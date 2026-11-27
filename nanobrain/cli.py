"""Plan is the default. Only --execute can contact an LLM or train real data."""

import argparse
import dataclasses
import json
from pathlib import Path

from .infra.contracts import CONFIG_PATH, DEFINITION, ROOT, load_config
from .infra.workflow import Workflow, plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="device for training and evaluation")
    ap.add_argument("--execute", action="store_true", help="explicitly start/resume real ReAct evolution")
    ap.add_argument("--freeze", action="store_true", help="lock the campaign definition, data and judges")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    # Workers use the committed canonical file; avoid a controller/worker config mismatch.
    if (
        args.config.resolve() != CONFIG_PATH.resolve()
        and json.loads(args.config.read_text()) != load_config()
    ):
        raise ValueError("workers require the committed canonical configuration")
    if args.freeze:
        from .infra.freeze import freeze

        lock = freeze(cfg)
        print(json.dumps({k: lock[k] for k in ("campaign_id", "frozen_at")} | {"files": len(lock["files"])}))
        return
    if not args.execute:
        print(json.dumps(plan(cfg, args.device), indent=2))
        return
    # Do not import/read credential configuration in plan or acceptance tests.
    from .agent.api_config import APIConfig
    from .agent.llm_client import OpenAICompatibleClient

    api = dataclasses.replace(
        APIConfig.from_env(env_file=cfg["api_env_file"]), model=cfg["model"], max_retries=-1, timeout_seconds=900
    )
    api.validate(require_credentials=True)
    output_directory = DEFINITION["campaign"].get("output_directory", "cajal-react-self-contained-20261110")
    if args.device == "cpu":
        output_directory += "-cpu"
    output = args.output or ROOT / "research-runs" / output_directory
    workflow = Workflow(cfg, output, device=args.device)
    try:
        workflow.execute(OpenAICompatibleClient(api))
    except Exception as exc:  # noqa: BLE001 - redact credentials at the controller boundary
        message = str(exc).replace(api.api_key, "[redacted]") if api.api_key else str(exc)
        raise RuntimeError(message) from None


if __name__ == "__main__":
    main()
