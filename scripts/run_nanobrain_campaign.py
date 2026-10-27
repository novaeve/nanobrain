#!/usr/bin/env python3
"""Run one parallel nanoBrain AutoResearch generation.

The fixed 100k-row subset and the existing ``train_nanobrain.py`` entry point
are reused for every candidate.  One independent candidate is placed on each
requested physical GPU; this is candidate parallelism rather than a second,
unverified distributed training implementation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.api_config import APIConfig  # noqa: E402
from nanobrain.campaign import (  # noqa: E402
    audit_training_receipt,
    baseline_regression_gate,
    candidate_method,
    genome_diff,
    metric_objectives,
    overall_metrics,
    primary_metric_record,
    read_metric_table,
    strongest_baselines,
)
from nanobrain.genome import CandidateGenome, SEARCH_SPACE, default_genome  # noqa: E402
from nanobrain.llm_client import APIRequestError, OpenAICompatibleClient  # noqa: E402
from nanobrain.search import pareto_front, score_receipt  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _prepare_output(requested: Path | None) -> Path:
    """Create a fresh campaign directory without mixing prior artifacts."""

    if requested is not None:
        output = requested.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"explicit output directory must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        return output

    base = ROOT / "results" / "nanobrain" / f"campaign-{_timestamp()}"
    output = base
    suffix = 1
    while output.exists():
        output = base.with_name(f"{base.name}-{suffix:02d}")
        suffix += 1
    output.mkdir(parents=True)
    return output


def _parse_gpus(value: str) -> list[int]:
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            gpu = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid GPU id: {item!r}") from exc
        if gpu < 0:
            raise argparse.ArgumentTypeError("GPU ids must be non-negative")
        values.append(gpu)
    if not values or len(values) > 8 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide one to eight unique GPU ids")
    return values


def _extract_json(text: str) -> Any:
    """Extract the first complete JSON value, tolerating markdown fences."""

    cleaned = re.sub(r"```(?:json)?", "", str(text), flags=re.IGNORECASE).replace("```", "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("API response did not contain a JSON object or array")


def _candidate_payloads(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("candidates", "genomes", "proposals"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("genome"), dict):
            item = item["genome"]
        if isinstance(item, dict):
            result.append(item)
    return result


def _evaluation_summary(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load compact, overall validation metrics keyed by candidate ID."""

    metric_allowlist = {
        "annotation": {"accuracy", "balanced_accuracy", "f1_macro"},
        "integration": {"integration_composite", "biology_conservation", "batch_removal"},
        "spatial_niche": {"ari", "nmi", "celltype_asw"},
        "spatial_completion": {"rmse", "heldout_rmse", "heldout_pearson"},
        "development": {"population_mse", "row_mse", "row_mae"},
    }
    summaries: dict[str, dict[str, Any]] = {}
    for path in paths:
        table = path / "benchmark_table.csv" if path.is_dir() else path
        if not table.is_file():
            raise ValueError(f"parent evaluation table does not exist: {table}")
        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            method = str(row.get("method", ""))
            prefix = "nanobrain_candidate-"
            if not method.startswith(prefix) or row.get("status") != "complete":
                continue
            if row.get("stratum_type") != "overall" or row.get("stratum_value") != "overall":
                continue
            task = str(row.get("task", ""))
            metric = str(row.get("metric", ""))
            if metric not in metric_allowlist.get(task, set()):
                continue
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            candidate_id = method.removeprefix(prefix)
            summary = summaries.setdefault(
                candidate_id,
                {"tasks": {}, "sources": []},
            )
            summary["tasks"].setdefault(task, {})[metric] = value
            source = str(table.resolve())
            if source not in summary["sources"]:
                summary["sources"].append(source)
    return summaries


def _history_summary(
    path: Path | None,
    evaluation_paths: list[Path] | None = None,
    parent_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read compact, non-sensitive results from a previous campaign."""

    if path is None:
        return []
    campaign_path = path / "campaign.json" if path.is_dir() else path
    if not campaign_path.is_file():
        raise ValueError(f"parent campaign does not exist: {campaign_path}")
    try:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read parent campaign: {campaign_path}") from exc
    evaluations = _evaluation_summary(evaluation_paths or [])
    summary: list[dict[str, Any]] = []
    training_steps = campaign.get("training", {}).get("steps")
    for candidate in campaign.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or (parent_candidate_id and candidate_id != parent_candidate_id):
            continue
        directory_value = candidate.get("directory")
        if directory_value:
            candidate_dir = Path(str(directory_value))
            if not candidate_dir.is_absolute():
                candidate_dir = campaign_path.parent / candidate_dir
        else:
            candidate_dir = campaign_path.parent / f"candidate_{int(candidate.get('index', -1)):02d}"
        receipt_candidates: list[Path] = []
        if isinstance(training_steps, int):
            receipt_candidates.append(candidate_dir / f"receipt-{candidate_id}-steps{training_steps}.json")
        receipt_candidates.extend(sorted(candidate_dir.glob("receipt-*.json")))
        receipt = None
        seen_paths: set[Path] = set()
        for receipt_path in receipt_candidates:
            if receipt_path in seen_paths or not receipt_path.is_file():
                continue
            seen_paths.add(receipt_path)
            try:
                candidate_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if candidate_receipt.get("candidate_id") != candidate_id:
                continue
            receipt = candidate_receipt
            break
        if not isinstance(receipt, dict):
            continue
        validation = receipt.get("validation", {})
        record: dict[str, Any] = {
            "candidate_id": receipt.get("candidate_id"),
            "genome": receipt.get("genome", candidate.get("genome", {})),
            "quality": receipt.get("quality"),
            "runtime_seconds": receipt.get("runtime_seconds"),
            "validation": {
                key: validation.get(key)
                for key in (
                    "annotation_accuracy",
                    "niche_accuracy",
                    "integration_accuracy",
                    "development_mse",
                    "completion_mse",
                )
                if key in validation
            },
        }
        if candidate_id in evaluations:
            record["benchmark_validation"] = evaluations[candidate_id]
        summary.append(record)
    return summary


def _proposal_prompt(
    count: int, seed: int, history: list[dict[str, Any]] | None = None
) -> str:
    space = {key: list(values) for key, values in SEARCH_SPACE.items()}
    candidate_shape = {key: f"<{key}>" for key in space}
    prompt = (
        "You are proposing candidates for a constrained nanoBrain research benchmark. "
        "Return JSON only, with exactly one candidates array and every listed field: "
        + json.dumps({"candidates": [candidate_shape]}, sort_keys=True)
        + ". "
        f"Provide {count} diverse valid candidates, seed them conceptually from {seed}, "
        "and change only fields listed in this search space. Do not include prose.\n"
        + json.dumps(space, sort_keys=True)
    )
    if history:
        prompt += (
            "\nA previous generation was evaluated with the same fixed data split. "
            "Use these results to propose a useful next generation; favor higher "
            "quality while keeping runtime and parameter count reasonable.\n"
            + json.dumps(history, sort_keys=True, ensure_ascii=True)
        )
    return prompt


def _api_proposals(
    config: APIConfig,
    count: int,
    seed: int,
    history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Ask the configured provider for candidates; return safe error text."""

    try:
        client = OpenAICompatibleClient(config)
        response = client.complete(
            _proposal_prompt(count, seed, history),
            temperature=0.2,
            max_tokens=2000,
        )
        payloads = _candidate_payloads(_extract_json(response))
        return payloads, response, None
    except (APIRequestError, ValueError, TypeError) as exc:
        # The client already retries transient errors indefinitely when
        # max_retries=-1.  Reaching this branch means a permanent provider
        # response or malformed proposal; local mutation keeps the campaign
        # reproducible instead of losing the GPU allocation.
        return [], None, f"{type(exc).__name__}: {exc}"


def _validated_genomes(
    proposals: list[dict[str, Any]], *, count: int, seed: int
) -> tuple[list[CandidateGenome], list[str]]:
    genomes: list[CandidateGenome] = []
    rejected: list[str] = []
    for payload in proposals:
        try:
            genomes.append(CandidateGenome.from_dict(payload))
        except (TypeError, ValueError) as exc:
            rejected.append(str(exc))
    rng = random.Random(int(seed))
    base = default_genome(seed=seed)
    while len(genomes) < count:
        if not genomes:
            candidate = base
        else:
            candidate = genomes[len(genomes) % len(genomes)].mutate(rng, rate=0.5)
        genomes.append(candidate.validate())
    # Every process gets a deterministic, distinct seed, including candidates
    # returned by the remote proposer.
    for index, candidate in enumerate(genomes[:count]):
        candidate.seed = int(seed + index)
        candidate.validate()
    return genomes[:count], rejected


def _launch_candidates(
    *,
    genomes: list[CandidateGenome],
    gpus: list[int],
    data: Path,
    output: Path,
    steps: int,
    batch_size: int,
    validation_batch_size: int,
    seed: int,
    precision: str = "bf16",
    development_cells: int = 64,
    validation_development_groups: int = 32,
    eval_interval: int = 0,
    loss_balance: str = "ema_equal",
    loss_balance_decay: float = 0.99,
    directory_names: list[str] | None = None,
    task: str = "joint",
    checkpoint_policy: str = "all",
    integration_adversarial_warmup_steps: int = 500,
    integration_logit_clip: float = 20.0,
    stability_max_loss: float = 1.0e8,
    stability_max_gradient_norm: float = 1.0e6,
) -> list[dict[str, Any]]:
    train_script = ROOT / "scripts" / "train_nanobrain.py"
    processes: list[dict[str, Any]] = []
    for index, (genome, gpu) in enumerate(zip(genomes, gpus, strict=True)):
        name = directory_names[index] if directory_names is not None else f"candidate_{index:02d}"
        candidate_dir = output / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        genome_path = candidate_dir / "genome.json"
        genome_path.write_text(
            json.dumps(genome.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_path = candidate_dir / "train.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(train_script),
            "--task", task,
            "--checkpoint-policy", checkpoint_policy,
            "--data",
            str(data),
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
            "--validation-batch-size",
            str(validation_batch_size),
            "--seed",
            str(genome.seed),
            "--device",
            "cuda",
            "--genome-json",
            str(genome_path),
            "--output",
            str(candidate_dir),
            "--precision",
            precision,
            "--development-cells",
            str(development_cells),
            "--validation-development-groups",
            str(validation_development_groups),
            "--eval-interval",
            str(eval_interval),
            "--loss-balance",
            loss_balance,
            "--loss-balance-decay",
            str(loss_balance_decay),
            "--integration-adversarial-warmup-steps",
            str(integration_adversarial_warmup_steps),
            "--integration-logit-clip",
            str(integration_logit_clip),
            "--stability-max-loss",
            str(stability_max_loss),
            "--stability-max-gradient-norm",
            str(stability_max_gradient_norm),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # These are candidate-local controls; callers can raise them when the
        # host has spare CPU capacity.
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", environment["OMP_NUM_THREADS"])
        environment.setdefault("OPENBLAS_NUM_THREADS", environment["OMP_NUM_THREADS"])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(
            {
                "index": index,
                "gpu": gpu,
                "pid": process.pid,
                "candidate_id": genome.genome_id,
                "genome": genome.to_dict(),
                "directory": str(candidate_dir.resolve()),
                "log": str(log_path),
                "process": process,
                "log_handle": log_handle,
                "started_at": time.time(),
            }
        )
    return processes


def _public_process(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"process", "log_handle", "started_at"}
    }


def _public_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"process", "log_handle", "started_at", "_training_receipt"}
    }


def _wait_processes(
    processes: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    phase: str,
) -> bool:
    while True:
        active = 0
        for item in processes:
            process = item["process"]
            code = process.poll()
            if code is None:
                active += 1
            elif "returncode" not in item:
                item["returncode"] = int(code)
                item["elapsed_seconds"] = float(time.time() - item["started_at"])
        manifest["status"] = f"{phase}_running" if active else f"{phase}_complete"
        manifest["active_processes"] = active
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        if not active:
            break
        time.sleep(2.0)
    for item in processes:
        item["log_handle"].close()
    return all(item.get("returncode") == 0 for item in processes)


def _load_training_receipt(item: dict[str, Any], steps: int) -> dict[str, Any] | None:
    path = Path(item["directory"]) / f"receipt-{item['candidate_id']}-steps{steps}.json"
    item["receipt"] = str(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("candidate_id") != item["candidate_id"]:
        return None
    return payload


BASELINE_METHODS = (
    "linear_sgd",
    "batch_centered_pca",
    "spatial_neighbor_mean",
    "gene_mean",
    "development_identity",
    "development_global_delta",
    "development_ridge_delta",
)


def _materialize_baseline_reference(
    *, data: Path, output: Path, seed: int
) -> Path:
    evaluator = ROOT / "scripts" / "run_nanobrain_baselines.py"
    baseline_output = output / "baseline_reference"
    baseline_output.mkdir(parents=True, exist_ok=False)
    log_path = baseline_output / "run.log"
    command = [
        sys.executable,
        str(evaluator),
        "--data",
        str(data),
        "--output",
        str(baseline_output),
        "--seed",
        str(seed),
        "--methods",
        ",".join(BASELINE_METHODS),
        "--skip-nanobrain",
        "--no-update-latest",
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", environment["OMP_NUM_THREADS"])
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    table = baseline_output / "benchmark_table.csv"
    if completed.returncode != 0 or not table.is_file():
        raise RuntimeError(f"canonical baseline evaluation failed; see {log_path}")
    return table


def _launch_evaluations(
    *,
    candidates: list[dict[str, Any]],
    gpus: list[int],
    data: Path,
    batch_size: int,
    evaluation_seed: int | None = None,
    evaluation_split: str = "validation",
    output_name: str = "evaluation",
) -> list[dict[str, Any]]:
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    evaluator = ROOT / "scripts" / "run_nanobrain_baselines.py"
    processes: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        audit = candidate.get("training_audit", {})
        checkpoint = audit.get("checkpoint")
        if not audit.get("passed") or not checkpoint:
            candidate["evaluation_returncode"] = None
            candidate["evaluation_error"] = "training_audit_failed"
            continue
        gpu = gpus[index % len(gpus)]
        output = Path(candidate["directory"]) / output_name
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "run.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(evaluator),
            "--data",
            str(data),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
            "--skip-baselines",
            "--no-update-latest",
        ]
        if evaluation_seed is not None:
            command.extend(["--population-evaluation-seed", str(evaluation_seed),
                            "--seed", str(evaluation_seed)])
        if evaluation_split == "test":
            command.extend(["--evaluation-split", "test", "--allow-test"])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", environment["OMP_NUM_THREADS"])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        candidate["evaluation"] = str(output / "benchmark_table.csv")
        processes.append(
            {
                "candidate": candidate,
                "candidate_id": candidate["candidate_id"],
                "gpu": gpu,
                "pid": process.pid,
                "log": str(log_path),
                "process": process,
                "log_handle": log_handle,
                "started_at": time.time(),
            }
        )
    return processes


def _finalize_evaluations(
    processes: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> bool:
    success = _wait_processes(
        processes, manifest=manifest, manifest_path=manifest_path, phase="evaluation"
    ) if processes else True
    for item in processes:
        candidate = item["candidate"]
        candidate["evaluation_returncode"] = item.get("returncode")
        candidate["evaluation_elapsed_seconds"] = item.get("elapsed_seconds")
        if item.get("returncode") != 0:
            candidate["evaluation_error"] = f"evaluator_exit_{item.get('returncode')}"
    return success


def _nearest_parent(
    genome: dict[str, Any], parents: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not parents:
        return None
    mutable = tuple(SEARCH_SPACE)
    return min(
        parents,
        key=lambda parent: (
            sum(parent["genome"].get(key) != genome.get(key) for key in mutable),
            str(parent["candidate_id"]),
        ),
    )


def _local_genomes(
    parents: list[dict[str, Any]], *, count: int, seed: int
) -> list[CandidateGenome]:
    rng = random.Random(int(seed))
    if not parents:
        base = default_genome(seed=seed)
        controls = [base.to_dict()]
        mlp = base.to_dict()
        mlp.update({"flow": "mlp", "pairing": "sinkhorn"})
        controls.append(mlp)
        no_flow = base.to_dict()
        no_flow.update({"flow": "none", "pairing": "identity"})
        controls.append(no_flow)
        genomes = [CandidateGenome.from_dict(value) for value in controls[:count]]
        while len(genomes) < count:
            genomes.append(base.mutate(rng, rate=0.45))
    else:
        parent_genomes = [CandidateGenome.from_dict(value["genome"]) for value in parents]
        genomes = []
        while len(genomes) < count:
            left = rng.choice(parent_genomes)
            right = rng.choice(parent_genomes)
            genomes.append(left.crossover(right, rng).mutate(rng, rate=0.35))
    for index, genome in enumerate(genomes):
        genome.seed = int(seed + index)
        genome.validate()
    return genomes


def _analyze_candidates(
    candidates: list[dict[str, Any]], baseline_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[str]]:
    baselines = strongest_baselines(baseline_rows)
    scored = []
    for candidate in candidates:
        promotion: dict[str, Any] = {
            "eligible": False,
            "reasons": [],
            "primary_metrics": {},
            "task_objectives": [],
        }
        audit = candidate.get("training_audit", {})
        promotion["reasons"].extend(audit.get("reasons", []))
        table = Path(str(candidate.get("evaluation", "")))
        metrics: dict[tuple[str, str], float] = {}
        if candidate.get("evaluation_returncode") != 0 or not table.is_file():
            promotion["reasons"].append("canonical_evaluation_incomplete")
        else:
            try:
                metrics = overall_metrics(
                    read_metric_table(table), candidate_method(candidate["candidate_id"])
                )
                objectives = metric_objectives(metrics)
                promotion["task_objectives"] = list(objectives)
                promotion["primary_metrics"] = primary_metric_record(metrics)
            except ValueError as exc:
                promotion["reasons"].append(f"metric_contract:{exc}")
        gate = baseline_regression_gate(metrics, baselines)
        promotion["baseline_gate"] = gate
        promotion["reasons"].extend(gate["reasons"])
        promotion["reasons"] = sorted(set(promotion["reasons"]))
        promotion["eligible"] = not promotion["reasons"]
        candidate["promotion"] = promotion
        receipt = candidate.get("_training_receipt")
        if isinstance(receipt, dict) and promotion["task_objectives"]:
            score_payload = dict(receipt)
            score_payload["promotion"] = promotion
            scored.append(score_receipt(score_payload))
    front = pareto_front(scored)
    pareto_ids = sorted(score.candidate_id for score in front)
    for candidate in candidates:
        candidate["promotion"]["pareto"] = candidate["candidate_id"] in pareto_ids
    return candidates, pareto_ids


def _aggregate_replicates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"passed": False, "reason": "no_replicates", "metrics": {}}
    metric_names = sorted(
        set.intersection(*[
            set(candidate.get("promotion", {}).get("primary_metrics", {}))
            for candidate in candidates
        ])
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [float(candidate["promotion"]["primary_metrics"][name]) for candidate in candidates]
        metrics[name] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    passed = (
        len(candidates) >= 3
        and len(metric_names) == 5
        and all(candidate.get("promotion", {}).get("eligible") for candidate in candidates)
    )
    return {"passed": passed, "metrics": metrics, "replicates": len(candidates)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("NANOBRAIN_DATA", ROOT / "assets" / "nanobrain_v3_100k")))
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("5,6,7"))
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=2048)
    parser.add_argument("--development-cells", type=int, default=64)
    parser.add_argument("--validation-development-groups", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--loss-balance",
        choices=("ema_equal", "detached_max_equal", "fixed"),
        default="ema_equal",
    )
    parser.add_argument("--loss-balance-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--parent-campaign",
        type=Path,
        default=None,
        help="optional prior campaign directory/receipt for API-guided evolution",
    )
    parser.add_argument(
        "--parent-evaluation",
        action="append",
        type=Path,
        default=[],
        help="benchmark directory/table whose overall validation metrics guide evolution; repeatable",
    )
    parser.add_argument(
        "--parent-candidate-id",
        default=None,
        help="optionally restrict proposal history to one evaluated parent candidate",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=None,
        help="existing canonical validation table; otherwise controls are fit once automatically",
    )
    parser.add_argument("--max-promotions", type=int, default=1)
    parser.add_argument("--replicate-seeds", type=int, default=3)
    parser.add_argument("--skip-replicates", action="store_true")
    parser.add_argument("--no-api", action="store_true", help="use deterministic local mutations only")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.validation_batch_size < 1
        or args.evaluation_batch_size < 1
        or args.generations < 1
        or args.max_promotions < 1
        or args.replicate_seeds < 1
        or not 0.0 <= args.loss_balance_decay < 1.0
    ):
        parser.error("steps, generations, promotions, replicate count, and batch sizes must be positive")
    if not args.skip_replicates and args.replicate_seeds > len(args.gpus):
        parser.error("--replicate-seeds cannot exceed the available GPU count")
    if not args.data.is_dir():
        parser.error(f"data directory does not exist: {args.data}")
    try:
        output = _prepare_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    api_config = APIConfig.from_env()
    manifest_path = output / "campaign.json"
    manifest: dict[str, Any] = {
        "format": "nanobrain_campaign_v2",
        "status": "preparing_baseline_reference",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()),
        "gpus": args.gpus,
        "training": {
            "generations": args.generations,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "development_cells": args.development_cells,
            "validation_development_groups": args.validation_development_groups,
            "eval_interval": args.eval_interval,
            "precision": args.precision,
            "loss_balance": args.loss_balance,
            "loss_balance_decay": args.loss_balance_decay,
            "seed": args.seed,
        },
        "api": api_config.redacted(),
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_opened": False,
        "candidates": [],
        "generations": [],
        "promotions": [],
    }
    _write_json(manifest_path, manifest)
    try:
        history = _history_summary(
            args.parent_campaign,
            evaluation_paths=args.parent_evaluation,
            parent_candidate_id=args.parent_candidate_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.parent_candidate_id and not history:
        parser.error(f"parent candidate was not found: {args.parent_candidate_id}")
    manifest.update(
        {
            "parent_campaign": str(args.parent_campaign.resolve()) if args.parent_campaign else None,
            "parent_evaluations": [str(path.resolve()) for path in args.parent_evaluation],
            "parent_candidate_id": args.parent_candidate_id,
            "parent_candidates": len(history),
        }
    )

    try:
        if args.baseline_reference is None:
            baseline_table = _materialize_baseline_reference(
                data=args.data.resolve(), output=output, seed=args.seed
            )
        else:
            baseline_table = (
                args.baseline_reference / "benchmark_table.csv"
                if args.baseline_reference.is_dir()
                else args.baseline_reference
            ).resolve()
        baseline_rows = read_metric_table(baseline_table)
        if len(strongest_baselines(baseline_rows)) != 5:
            raise ValueError("baseline reference does not cover every primary task metric")
    except (OSError, RuntimeError, ValueError) as exc:
        manifest["status"] = "baseline_reference_failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        return 1
    manifest["baseline_reference"] = str(baseline_table)
    _write_json(manifest_path, manifest)

    parent_pool = [
        {"candidate_id": item["candidate_id"], "genome": item["genome"]}
        for item in history
        if isinstance(item.get("genome"), dict)
    ]
    proposal_history = list(history)
    all_candidates: list[dict[str, Any]] = []
    final_generation: list[dict[str, Any]] = []
    global_index = 0

    for generation in range(args.generations):
        generation_seed = int(args.seed + generation * 10_000)
        manifest["status"] = "waiting_for_api" if not args.no_api else "preparing_candidates"
        manifest["current_generation"] = generation
        _write_json(manifest_path, manifest)
        proposals: list[dict[str, Any]] = []
        api_response: str | None = None
        api_error: str | None = None
        if not args.no_api:
            proposals, api_response, api_error = _api_proposals(
                api_config, len(args.gpus), generation_seed, proposal_history
            )
            if api_response is not None:
                (output / f"api_proposal_generation_{generation:02d}.txt").write_text(
                    api_response, encoding="utf-8"
                )
        if proposals:
            genomes, rejected = _validated_genomes(
                proposals, count=len(args.gpus), seed=generation_seed
            )
        else:
            genomes = _local_genomes(parent_pool, count=len(args.gpus), seed=generation_seed)
            rejected = []

        generation_dir = output / f"generation_{generation:02d}"
        generation_dir.mkdir(parents=True, exist_ok=False)
        processes = _launch_candidates(
            genomes=genomes,
            gpus=args.gpus,
            data=args.data.resolve(),
            output=generation_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            seed=generation_seed,
            precision=args.precision,
            development_cells=args.development_cells,
            validation_development_groups=args.validation_development_groups,
            eval_interval=args.eval_interval,
            loss_balance=args.loss_balance,
            loss_balance_decay=args.loss_balance_decay,
        )
        current: list[dict[str, Any]] = []
        for item in processes:
            parent = _nearest_parent(item["genome"], parent_pool)
            item["index"] = global_index
            item["generation"] = generation
            item["parent_candidate_id"] = parent["candidate_id"] if parent else None
            item["mutation_diff"] = genome_diff(parent["genome"] if parent else None, item["genome"])
            global_index += 1
            current.append(item)
        all_candidates.extend(current)
        manifest["candidates"] = [_public_candidate(item) for item in all_candidates]
        manifest["generations"].append(
            {
                "generation": generation,
                "seed": generation_seed,
                "api_proposals_received": len(proposals),
                "api_proposals_rejected": rejected,
                "api_error": api_error,
                "candidate_ids": [item["candidate_id"] for item in current],
                "pareto_candidate_ids": [],
            }
        )
        manifest["status"] = "training_launched"
        _write_json(manifest_path, manifest)
        print(json.dumps({
            "campaign": str(manifest_path),
            "generation": generation,
            "pids": [item["pid"] for item in processes],
            "gpus": args.gpus,
        }, indent=2))
        if not args.wait:
            for item in processes:
                item["log_handle"].close()
            manifest["status"] = "detached_training"
            manifest["candidates"] = [_public_candidate(item) for item in all_candidates]
            _write_json(manifest_path, manifest)
            return 0

        _wait_processes(
            processes,
            manifest=manifest,
            manifest_path=manifest_path,
            phase=f"generation_{generation:02d}_training",
        )
        for item in current:
            receipt = _load_training_receipt(item, args.steps)
            if receipt is None:
                receipt = {}
            item["_training_receipt"] = receipt
            audit = audit_training_receipt(receipt)
            if item.get("returncode") != 0:
                audit["passed"] = False
                audit["reasons"] = sorted(set(audit["reasons"] + ["training_process_failed"]))
            item["training_audit"] = audit
            item["receipt_summary"] = {
                key: receipt.get(key)
                for key in ("quality", "best_validation_quality", "runtime_seconds", "parameter_count")
            }

        evaluations = _launch_evaluations(
            candidates=current,
            gpus=args.gpus,
            data=args.data.resolve(),
            batch_size=args.evaluation_batch_size,
        )
        _finalize_evaluations(
            evaluations, manifest=manifest, manifest_path=manifest_path
        )
        current, pareto_ids = _analyze_candidates(current, baseline_rows)
        manifest["generations"][-1]["pareto_candidate_ids"] = pareto_ids
        manifest["candidates"] = [_public_candidate(item) for item in all_candidates]
        _write_json(manifest_path, manifest)
        final_generation = current

        parent_pool = [
            {"candidate_id": item["candidate_id"], "genome": item["genome"]}
            for item in current
            if item["candidate_id"] in pareto_ids
        ]
        proposal_history = [
            {
                "candidate_id": item["candidate_id"],
                "genome": item["genome"],
                "quality": item["receipt_summary"].get("quality"),
                "runtime_seconds": item["receipt_summary"].get("runtime_seconds"),
                "benchmark_validation": {
                    "primary_metrics": item["promotion"]["primary_metrics"]
                },
            }
            for item in current
            if item["promotion"]["task_objectives"]
        ]
        if generation + 1 < args.generations and not parent_pool:
            manifest["status"] = "evolution_failed_no_valid_parent"
            manifest["candidates"] = [_public_candidate(item) for item in all_candidates]
            _write_json(manifest_path, manifest)
            return 1

    promotable = [
        item
        for item in final_generation
        if item["promotion"].get("eligible") and item["promotion"].get("pareto")
    ]
    promotable.sort(
        key=lambda item: (
            float(item.get("receipt_summary", {}).get("quality") or -math.inf),
            -float(item.get("receipt_summary", {}).get("runtime_seconds") or math.inf),
            item["candidate_id"],
        ),
        reverse=True,
    )
    selected = promotable[: args.max_promotions]
    selected_ids = {item["candidate_id"] for item in selected}
    for item in final_generation:
        item["promotion"]["selected"] = item["candidate_id"] in selected_ids
        item["promotion"]["selection_reason"] = (
            "final_generation_pareto_and_all_baseline_gates_passed"
            if item["candidate_id"] in selected_ids
            else "not_selected_by_fail_closed_promotion"
        )

    if not args.skip_replicates:
        for promotion_index, parent in enumerate(selected):
            base_seed = int(args.seed + 1_000_000 + promotion_index * 1_000)
            genomes = []
            for replicate_index in range(args.replicate_seeds):
                payload = dict(parent["genome"])
                payload["seed"] = base_seed + replicate_index
                genomes.append(CandidateGenome.from_dict(payload))
            promotion_dir = output / "promotions" / parent["candidate_id"]
            promotion_dir.mkdir(parents=True, exist_ok=False)
            names = [f"seed_{genome.seed}" for genome in genomes]
            processes = _launch_candidates(
                genomes=genomes,
                gpus=args.gpus[: args.replicate_seeds],
                data=args.data.resolve(),
                output=promotion_dir,
                steps=args.steps,
                batch_size=args.batch_size,
                validation_batch_size=args.validation_batch_size,
                seed=base_seed,
                precision=args.precision,
                development_cells=args.development_cells,
                validation_development_groups=args.validation_development_groups,
                eval_interval=args.eval_interval,
                loss_balance=args.loss_balance,
                loss_balance_decay=args.loss_balance_decay,
                directory_names=names,
            )
            replicates: list[dict[str, Any]] = []
            for item in processes:
                item["replicate_of"] = parent["candidate_id"]
                item["parent_candidate_id"] = parent["candidate_id"]
                item["mutation_diff"] = genome_diff(parent["genome"], item["genome"])
                replicates.append(item)
            _wait_processes(
                processes,
                manifest=manifest,
                manifest_path=manifest_path,
                phase=f"replicates_{parent['candidate_id']}_training",
            )
            for item in replicates:
                receipt = _load_training_receipt(item, args.steps) or {}
                item["_training_receipt"] = receipt
                audit = audit_training_receipt(receipt)
                if item.get("returncode") != 0:
                    audit["passed"] = False
                    audit["reasons"] = sorted(set(audit["reasons"] + ["training_process_failed"]))
                item["training_audit"] = audit
                item["receipt_summary"] = {
                    key: receipt.get(key)
                    for key in ("quality", "best_validation_quality", "runtime_seconds", "parameter_count")
                }
            evaluations = _launch_evaluations(
                candidates=replicates,
                gpus=args.gpus[: args.replicate_seeds],
                data=args.data.resolve(),
                batch_size=args.evaluation_batch_size,
            )
            _finalize_evaluations(
                evaluations, manifest=manifest, manifest_path=manifest_path
            )
            replicates, _ = _analyze_candidates(replicates, baseline_rows)
            aggregate = _aggregate_replicates(replicates)
            manifest["promotions"].append(
                {
                    "candidate_id": parent["candidate_id"],
                    "replicates": [_public_candidate(item) for item in replicates],
                    "aggregate": aggregate,
                }
            )

    manifest["candidates"] = [_public_candidate(item) for item in all_candidates]
    manifest["selected_candidate_ids"] = sorted(selected_ids)
    manifest["scientific_promotion_passed"] = bool(selected) and (
        args.skip_replicates
        or all(item["aggregate"].get("passed") for item in manifest["promotions"])
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["test_opened"] = False
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "campaign": str(manifest_path),
        "status": manifest["status"],
        "selected_candidate_ids": manifest["selected_candidate_ids"],
        "scientific_promotion_passed": manifest["scientific_promotion_passed"],
        "test_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
