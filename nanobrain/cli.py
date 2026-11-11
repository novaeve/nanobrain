"""Plan is the default. Only --execute can contact an LLM or train real data."""

import argparse
import dataclasses
import json
from pathlib import Path

from .infra.contracts import ROOT, load_config
from .infra.workflow import Workflow, plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cajal_react.json")
    ap.add_argument("--output", type=Path, default=ROOT / "research-runs/cajal-react-self-contained-20261110")
    ap.add_argument("--execute", action="store_true", help="explicitly start/resume real ReAct evolution")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    # Workers use the committed canonical file; avoid a controller/worker config mismatch.
    if (
        args.config.resolve() != (ROOT / "configs/cajal_react.json").resolve()
        and json.loads(args.config.read_text()) != load_config()
    ):
        raise ValueError("workers require the committed canonical configuration")
    if not args.execute:
        print(json.dumps(plan(cfg), indent=2))
        return
    # Do not import/read credential configuration in plan or acceptance tests.
    from .agent.api_config import APIConfig
    from .agent.llm_client import OpenAICompatibleClient

    api = dataclasses.replace(
        APIConfig.from_env(env_file=cfg["api_env_file"]), model=cfg["model"], max_retries=-1, timeout_seconds=900
    )
    api.validate(require_credentials=True)
    workflow = Workflow(cfg, args.output)
    try:
        workflow.execute(OpenAICompatibleClient(api))
    except Exception as exc:  # noqa: BLE001 - redact credentials at the controller boundary
        message = str(exc).replace(api.api_key, "[redacted]") if api.api_key else str(exc)
        raise RuntimeError(message) from None


if __name__ == "__main__":
    main()
