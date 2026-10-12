"""Canonical task-by-metric Experimental Results contract.

This is independent from the historical v1 registry: v1 ranks many
experiment-specific benchmarks, while v2 binds every task to one global split.
"""

from __future__ import annotations

import json
import fcntl
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

CONTRACT_FORMAT = "brainwm_experimental_results_contract_v2"
RESULT_FORMAT = "brainwm_experimental_result_v2"
TASK_STATUS = {"ready", "proxy_only", "protocol_pending", "blocked_data"}
METHOD_CLASS = {"baseline", "brainwm", "ablation"}
PROTOCOLS = {"zero_shot", "few_shot", "supervised", "transductive_unsupervised"}
RESULT_STATUS = {"complete", "failed"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_contract(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(payload, project_root=path.resolve().parents[2])
    return payload


def validate_contract(
    contract: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    check_paths: bool = False,
) -> dict[str, Any]:
    """Validate the shared data boundary and all task table definitions."""

    root = Path(project_root or ".").resolve()
    _require(isinstance(contract, Mapping), "contract must be a mapping")
    _require(contract.get("format") == CONTRACT_FORMAT, "unsupported v2 contract")
    _require(bool(contract.get("contract_id")), "contract_id is required")
    data = contract.get("data")
    _require(isinstance(data, Mapping), "data contract is required")
    for field in (
        "data_contract_id", "label_contract_id", "artifact_dir",
        "task_labels_manifest", "split_id", "species",
    ):
        _require(data.get(field) not in (None, "", []), f"data.{field} is required")
    _require(data.get("species") == ["human"], "v2 formal benchmark is human-only")
    _require(
        data.get("fractions") == {"train": 0.6, "validation": 0.2, "test": 0.2},
        "global split must remain 6:2:2",
    )
    _require(data.get("same_split_for_all_tasks") is True, "tasks must share one split")
    _require(data.get("test_state") == "sealed", "global test must remain sealed")

    task_order = contract.get("task_order")
    tasks = contract.get("tasks")
    _require(isinstance(tasks, Mapping) and tasks, "tasks are required")
    _require(
        isinstance(task_order, list) and task_order == list(tasks),
        "task_order must exactly match task declarations",
    )
    _require(len(task_order) == len(set(task_order)), "task_order repeats a task")
    evaluators: set[str] = set()
    for task_id, task in tasks.items():
        _require(isinstance(task, Mapping), f"task {task_id} must be a mapping")
        _require(task.get("task_id") == task_id, f"task_id mismatch: {task_id}")
        _require(task.get("status") in TASK_STATUS, f"invalid task status: {task_id}")
        evaluator_id = task.get("evaluator_id")
        _require(isinstance(evaluator_id, str) and evaluator_id, f"evaluator missing: {task_id}")
        _require(evaluator_id not in evaluators, f"evaluator reused: {evaluator_id}")
        evaluators.add(evaluator_id)
        _require(task.get("split_id") == data["split_id"], f"split mismatch: {task_id}")
        metrics = task.get("metrics")
        _require(isinstance(metrics, Mapping), f"metrics invalid: {task_id}")
        if task.get("status") == "blocked_data":
            _require(not metrics, f"blocked task cannot claim metrics: {task_id}")
        else:
            _require(bool(metrics), f"active task needs metrics: {task_id}")
        primary = task.get("primary_metric")
        if primary is not None:
            _require(primary in metrics, f"unknown primary metric: {task_id}")
        for metric_id, metric in metrics.items():
            _require(isinstance(metric, Mapping), f"invalid metric: {task_id}.{metric_id}")
            _require(metric.get("direction") in {"maximize", "minimize"}, f"invalid direction: {metric_id}")
            _require(bool(metric.get("display_name")), f"metric display name missing: {metric_id}")
        rows = task.get("rows")
        _require(isinstance(rows, list) and rows, f"method rows missing: {task_id}")
        seen: set[tuple[str, str]] = set()
        for row in rows:
            _require(isinstance(row, Mapping), f"method row invalid: {task_id}")
            _require(row.get("method_class") in METHOD_CLASS, f"method class invalid: {task_id}")
            protocol = row.get("training_protocol")
            _require(protocol in PROTOCOLS, f"training protocol invalid: {task_id}")
            key = (str(row.get("method_id", "")), str(protocol))
            _require(key[0] and key not in seen, f"duplicate method/protocol: {task_id}.{key}")
            seen.add(key)
            if protocol == "few_shot":
                _require(int(row.get("shots", 0)) > 0, f"few-shot row lacks shots: {task_id}.{key[0]}")
            else:
                _require(row.get("shots") in (None, 0), f"shots only apply to few-shot: {task_id}.{key[0]}")

    if check_paths:
        artifact = root / str(data["artifact_dir"])
        manifest_path = artifact / "manifest.json"
        access_path = artifact / "access_policy.json"
        _require(artifact.is_dir(), f"missing benchmark artifact: {artifact}")
        _require(manifest_path.is_file() and access_path.is_file(), "global manifests are missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        access = json.loads(access_path.read_text(encoding="utf-8"))
        _require(manifest.get("split_id") == data["split_id"], "artifact split drifted")
        _require(manifest.get("data_contract_id") == data["data_contract_id"], "artifact data contract drifted")
        _require(access.get("sealed") is True, "artifact test policy is not sealed")
        labels_path = root / str(data["task_labels_manifest"])
        _require(labels_path.is_file(), f"missing task-label manifest: {labels_path}")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        _require(labels.get("format") == "brainwm_global_task_labels_v1", "unsupported task-label manifest")
        _require(labels.get("label_contract_id") == data["label_contract_id"], "label contract drifted")
        _require(labels.get("split_id") == data["split_id"], "task-label split drifted")
        _require(labels.get("data_contract_id") == data["data_contract_id"], "task-label data contract drifted")
        _require(set(labels.get("tasks", {})) == set(contract["task_order"]), "task-label tasks drifted")
        _require(labels.get("zero_copy_cell_labels") is True, "task labels must remain zero-copy")
        _require(labels["tasks"]["pseudotime"].get("ground_truth_available") is False, "pseudotime ground truth is unavailable")
        _require(labels["tasks"]["spatial_niche"].get("niche_category_ground_truth_available") is False, "niche category ground truth is unavailable")
        _require(labels["tasks"]["perturbation"].get("label_available") is False, "perturbation labels are unavailable")
    return {
        "valid": True,
        "contract_id": contract["contract_id"],
        "data_contract_id": data["data_contract_id"],
        "label_contract_id": data["label_contract_id"],
        "split_id": data["split_id"],
        "tasks": list(task_order),
    }


def load_results(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        _require(isinstance(value, dict), f"result line {line_number} must be an object")
        rows.append(value)
    return rows


def _method_rows(task: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["method_id"]), str(row["training_protocol"])): row
        for row in task["rows"]
    }


def validate_results(
    contract: Mapping[str, Any],
    results: Iterable[Mapping[str, Any]],
    *,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Validate comparable result rows against the global contract."""

    validate_contract(contract)
    root = Path(project_root).resolve() if project_root is not None else None
    data = contract["data"]
    tasks = contract["tasks"]
    checked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_rows: set[tuple[str, str, str]] = set()
    required = {
        "format", "result_id", "task_id", "method_id", "method_name",
        "method_class", "training_protocol", "data_contract_id", "split_id",
        "evaluator_id", "metrics", "status", "evidence_path", "label_access",
    }
    for result in results:
        _require(isinstance(result, Mapping), "each result must be a mapping")
        missing = required - set(result)
        _require(not missing, f"result missing fields: {sorted(missing)}")
        _require(result["format"] == RESULT_FORMAT, f"unsupported result: {result.get('result_id')}")
        result_id = str(result["result_id"])
        _require(result_id and result_id not in seen_ids, f"duplicate result_id: {result_id}")
        seen_ids.add(result_id)
        task_id = str(result["task_id"])
        _require(task_id in tasks, f"unknown task: {task_id}")
        task = tasks[task_id]
        _require(task["status"] not in {"blocked_data", "protocol_pending"}, f"task not result-ready: {task_id}")
        protocol = str(result["training_protocol"])
        method_key = (str(result["method_id"]), protocol)
        spec = _method_rows(task).get(method_key)
        _require(spec is not None, f"undeclared method/protocol: {task_id}.{method_key}")
        _require(result["method_name"] == spec["method_name"], f"method name drift: {result_id}")
        _require(result["method_class"] == spec["method_class"], f"method class drift: {result_id}")
        row_key = (task_id, method_key[0], protocol)
        _require(row_key not in seen_rows, f"duplicate table row: {row_key}")
        seen_rows.add(row_key)
        _require(result["data_contract_id"] == data["data_contract_id"], f"data contract mismatch: {result_id}")
        _require(result["split_id"] == data["split_id"], f"split mismatch: {result_id}")
        _require(result["evaluator_id"] == task["evaluator_id"], f"evaluator mismatch: {result_id}")
        _require(result["status"] in RESULT_STATUS, f"invalid result status: {result_id}")
        label_access = result["label_access"]
        _require(isinstance(label_access, Mapping), f"label_access invalid: {result_id}")
        for field in ("target_labels_used_for_fit", "target_labels_used_for_selection"):
            _require(isinstance(label_access.get(field), bool), f"label_access.{field} missing: {result_id}")
        if protocol in {"zero_shot", "transductive_unsupervised"}:
            _require(not label_access["target_labels_used_for_fit"], f"{protocol} used target fit labels: {result_id}")
            _require(not label_access["target_labels_used_for_selection"], f"{protocol} used target selection labels: {result_id}")
        if protocol == "few_shot":
            _require(int(result.get("shots", 0)) == int(spec["shots"]), f"few-shot budget mismatch: {result_id}")
        metrics = result["metrics"]
        _require(isinstance(metrics, Mapping), f"metrics invalid: {result_id}")
        declared = set(task["metrics"])
        _require(set(metrics) <= declared, f"undeclared metrics: {result_id}")
        required_metrics = {
            metric_id for metric_id, metric in task["metrics"].items()
            if metric.get("required", False)
        }
        if result["status"] == "complete":
            _require(required_metrics <= set(metrics), f"required metrics missing: {result_id}")
        _require(all(_finite(value) for value in metrics.values()), f"metrics must be finite: {result_id}")
        evidence = result["evidence_path"]
        _require(isinstance(evidence, str) and evidence and not Path(evidence).is_absolute(), f"evidence path invalid: {result_id}")
        if root is not None:
            evidence_path = (root / evidence).resolve()
            try:
                evidence_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"evidence outside project: {result_id}") from exc
            _require(evidence_path.is_file(), f"evidence missing: {result_id}")
        checked.append(dict(result))
    return checked


def append_result(
    path: str | Path,
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Append one validated result without modifying historical stores."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reload and validate while holding the same file lock used for the write;
    # concurrent GPU jobs must not append duplicate or partially interleaved
    # receipts to the shared experimental table.
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = load_results(path)
            checked = validate_results(
                contract, [*existing, result], project_root=project_root
            )
            normalized = checked[-1]
            handle.seek(0, 2)
            if existing and handle.tell() > 0:
                with path.open("rb") as tail:
                    tail.seek(-1, 2)
                    if tail.read(1) != b"\n":
                        handle.write("\n")
            handle.write(
                json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return normalized


def _value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(contract: Mapping[str, Any], results: Iterable[Mapping[str, Any]]) -> str:
    """Render one method-by-metric table for every task."""

    validate_contract(contract)
    by_key = {
        (str(row["task_id"]), str(row["method_id"]), str(row["training_protocol"])): row
        for row in results
    }
    data = contract["data"]
    lines = [
        "# BrainWM Experimental Results V2",
        "",
        f"所有任务共享 {data['split_id']}（train/validation/test = 6:2:2）和 sealed test 边界。每个任务单独排名；不同任务的 metric 不做横向合并。",
        "",
        "Zero-shot 表示目标测试标签不参与拟合或选择；Few-shot 必须报告固定 shots；Transductive 允许无标签目标表达参与预声明校正，但目标标签仍只用于最终评分。",
        "",
        "当前旧 v1 数值不迁入本表，因为旧 benchmark 与全局 V2 的 split/evaluator 不同。",
    ]
    for task_id in contract["task_order"]:
        task = contract["tasks"][task_id]
        lines.extend(["", f"## {task['display_name']}", "", task["objective"]])
        if task.get("note"):
            lines.extend(["", str(task["note"])])
        metrics = task["metrics"]
        headers = ["Method", "Class", "Protocol", *[m["display_name"] for m in metrics.values()], "Status"]
        lines.extend([
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ])
        for spec in task["rows"]:
            key = (task_id, str(spec["method_id"]), str(spec["training_protocol"]))
            result = by_key.get(key)
            protocol = str(spec["training_protocol"])
            if protocol == "few_shot":
                protocol = f"few-shot ({spec['shots']})"
            if task["status"] == "blocked_data" or spec.get("applicable") is False:
                metric_values = ["N/A"] * len(metrics)
                reason = spec.get("reason", "not applicable")
                status = f"N/A ({reason})"
            elif result is None:
                metric_values = ["—"] * len(metrics)
                status = "pending"
            else:
                metric_values = [_value(result["metrics"].get(metric_id)) for metric_id in metrics]
                status = str(result["status"])
                provenance = result.get("provenance")
                if isinstance(provenance, Mapping):
                    scope = provenance.get("scope")
                    if scope and scope != "global_validation_or_test":
                        status = f"{status} ({scope})"
            values = [str(spec["method_name"]), str(spec["method_class"]), protocol, *metric_values, status]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Registration rules",
        "",
        "新结果只能追加到 experimental_results_v2.jsonl，并必须匹配固定的 data_contract_id、split_id、任务 evaluator、方法协议和证据路径。测试集在模型、超参数和输出全部冻结后才允许一次性评分。",
    ])
    return "\n".join(lines) + "\n"
