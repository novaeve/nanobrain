"""Frozen label-support policy for the project-native integration benchmark.

The policy is derived from the pre-model label-count audit for the current
PsychAD donor split.  It deliberately separates labels that can contribute to
the biology score from labels that are retained for descriptive reporting only.
Do not change these IDs after model results are available; a changed cohort or
split requires a new policy revision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

POLICY_ID = "INT-LABEL-SUPPORT-23-INTERSECTION-V1"
MIN_CELLS_PER_LABEL = 20

# The donor-capped materialized evaluator manifest contains only 9 cells for
# CL:4023041 in each evaluation role.  V2 freezes the supported intersection
# of that actual manifest instead of inheriting counts from the uncapped donor
# split.  V1 remains immutable for provenance.
MATERIALIZED_POLICY_ID = "INT-LABEL-SUPPORT-22-MATERIALIZED-V2"

# These are the 23 labels with at least MIN_CELLS_PER_LABEL cells in both
# source_validation and target_test in the 2026-09-01 pre-model audit.
SCORED_LABELS: tuple[str, ...] = (
    "CL:0000084",
    "CL:0000115",
    "CL:0000127",
    "CL:0000128",
    "CL:0000129",
    "CL:0000192",
    "CL:0000617",
    "CL:0000623",
    "CL:0000669",
    "CL:0000881",
    "CL:0002453",
    "CL:4023040",
    "CL:4023041",
    "CL:4023042",
    "CL:4023046",
    "CL:4023051",
    "CL:4030059",
    "CL:4030065",
    "CL:4030067",
    "CL:4072027",
    "CL:4072029",
    "CL:4072032",
    "CL:4072034",
)

DESCRIPTIVE_ONLY_LABELS: tuple[str, ...] = (
    "CL:0000236",
    "CL:0000786",
)

MATERIALIZED_SCORED_LABELS: tuple[str, ...] = tuple(
    label for label in SCORED_LABELS if label != "CL:4023041"
)
MATERIALIZED_DESCRIPTIVE_ONLY_LABELS: tuple[str, ...] = (
    *DESCRIPTIVE_ONLY_LABELS,
    "CL:4023041",
)

DECLARED_LABELS = frozenset((*SCORED_LABELS, *DESCRIPTIVE_ONLY_LABELS))

_POLICY_DEFINITIONS = {
    POLICY_ID: (SCORED_LABELS, DESCRIPTIVE_ONLY_LABELS),
    MATERIALIZED_POLICY_ID: (
        MATERIALIZED_SCORED_LABELS,
        MATERIALIZED_DESCRIPTIVE_ONLY_LABELS,
    ),
}


def _counts(labels: Mapping[str, Any] | None, role: str) -> dict[str, int]:
    if not isinstance(labels, Mapping):
        return {}
    role_payload = labels.get(role, {})
    if not isinstance(role_payload, Mapping):
        return {}
    support = role_payload.get("label_support", {})
    if not isinstance(support, Mapping):
        return {}
    return {str(label): int(value) for label, value in support.items()}


def build_policy_snapshot(
    labels: Mapping[str, Any] | None,
    *,
    threshold: int = MIN_CELLS_PER_LABEL,
    policy_id: str = POLICY_ID,
) -> dict[str, Any]:
    """Return a reproducible policy snapshot and pre-model drift checks.

    The declared label IDs are intentionally fixed.  Counts are copied from
    the read-only audit for provenance; they never depend on an embedding or a
    model score.  ``status`` is ``frozen`` only when the observed evaluation
    labels exactly match this revision and all scored labels meet threshold.
    """

    if policy_id not in _POLICY_DEFINITIONS:
        raise ValueError(f"unknown integration label policy: {policy_id}")
    scored_labels, descriptive_only_labels = _POLICY_DEFINITIONS[policy_id]
    declared_labels = frozenset((*scored_labels, *descriptive_only_labels))
    source = _counts(labels, "source_validation")
    target = _counts(labels, "target_test")
    observed = set(source) | set(target)
    missing_declared = sorted(declared_labels - observed)
    unexpected = sorted(observed - declared_labels)
    support_failures: dict[str, dict[str, int]] = {}
    for role, counts in (("source_validation", source), ("target_test", target)):
        failed = {
            label: int(counts.get(label, 0))
            for label in scored_labels
            if int(counts.get(label, 0)) < int(threshold)
        }
        if failed:
            support_failures[role] = dict(sorted(failed.items()))

    return {
        "policy_id": policy_id,
        "mode": "supported_shared_intersection",
        "threshold": int(threshold),
        "required_partitions": ["source_validation", "target_test"],
        "scored_labels": list(scored_labels),
        "descriptive_only_labels": list(descriptive_only_labels),
        "selection_basis": (
            "Cell Ontology support counts in the fixed evaluation manifest; "
            f"{len(scored_labels)} labels meet threshold in both evaluation partitions"
        ),
        "source_validation_support": {
            label: int(source.get(label, 0))
            for label in (*scored_labels, *descriptive_only_labels)
        },
        "target_test_support": {
            label: int(target.get(label, 0))
            for label in (*scored_labels, *descriptive_only_labels)
        },
        "missing_declared_labels": missing_declared,
        "unexpected_evaluation_labels": unexpected,
        "support_failures": support_failures,
        "status": (
            "frozen"
            if labels is not None
            and not missing_declared
            and not unexpected
            and not support_failures
            and int(threshold) == MIN_CELLS_PER_LABEL
            else "blocked"
        ),
        "descriptive_only_in_primary_score": False,
        "batch_metrics_include_descriptive_only": True,
    }


def policy_is_frozen(policy: Mapping[str, Any] | None) -> bool:
    """Check the immutable structural fields of this policy revision."""

    if not isinstance(policy, Mapping):
        return False
    try:
        threshold = int(policy.get("threshold", -1))
    except (TypeError, ValueError):
        return False
    definition = _POLICY_DEFINITIONS.get(str(policy.get("policy_id", "")))
    if definition is None:
        return False
    scored_labels, descriptive_only_labels = definition
    return (
        policy.get("mode") == "supported_shared_intersection"
        and threshold == MIN_CELLS_PER_LABEL
        and list(policy.get("scored_labels", [])) == list(scored_labels)
        and list(policy.get("descriptive_only_labels", []))
        == list(descriptive_only_labels)
        and policy.get("descriptive_only_in_primary_score") is False
        and policy.get("batch_metrics_include_descriptive_only") is True
    )
