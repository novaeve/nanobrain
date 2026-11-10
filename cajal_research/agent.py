"""Adapted from nanoBrain ReAct: inspect → write → sandbox-validate → submit."""

import json
from contextlib import nullcontext

from .contracts import (
    BALANCER_RANGES,
    DATA_RANGES,
    KINDS,
    METRICS,
    MODEL_RANGES,
    MODEL_SPACE,
    OPTIMIZER_RANGES,
    TRAIN_INTS,
    TRAIN_RANGES,
    candidate_schema,
    save,
    semantic_id,
    validate_bundle,
)

SYSTEM = """Evolve the existing Cajal v1 shared encoder and its FIVE JOINT pretraining objectives.
Start from the actual frozen Cajal v1 architecture/configuration, not its trained weights,
a nanoBrain graph or independent task models. Every new candidate trains from random initialization.
Each candidate has ONE set of weights per seed. You may change allowed Cajal architecture/training
fields and add tensor-only PyTorch token modules, gene/classification losses or an optimizer.
Candidate model/train/data/components values are complete overrides relative to the frozen Cajal v1, not patches
relative to a parent: retain every desired ancestral change explicitly, including plugin sources.
The current Cajal unified_1m data, row identities, gene vocabularies, measurement panels, masks,
label hierarchies, five active objectives, seed 0, 30k LR schedule stopped at 10k and official
evaluation code are immutable. Allowed data fields tune training minibatch/sampling sizes only;
they never select an old 100k subset. The current campaign uses ONE seed, not three. Its scores
are single-seed exploratory evidence; do not invent a standard deviation or significance.
Use the merged component catalogue to combine original nanoBrain parts and newly split Cajal,
scVI and scANVI parts. Respect declared compatible slots and availability; never substitute a
different implementation for an unavailable source. Whole-method baseline scores do not transfer
automatically to a component hybrid. Do not use legacy nanoBrain experiment scores as evidence or
comparison thresholds: those experiments used different data and evaluation protocols. Reuse legacy
component implementations, but use only Cajal-compatible performance results in the LLM context.
Reuse existing Cajal baseline scores and the saved cajal_v1@10000 comparison reference; never refit
or rescore historical methods. Those artifacts are comparison evidence only. Ancestry transfers
architecture, source and hyperparameters, NEVER trained weights, predictions or teacher features.
No historical or externally pretrained checkpoint may initialize, supervise or be combined with
a candidate. Historical teacher_gram distillation is unavailable. Only a candidate's own interrupted
training may resume its same-seed checkpoint. The deliverable is a self-contained architecture and
fully specified training recipe reproducible from the declared training data and random seed.
Self-supervised losses and training schedules remain searchable within the fixed training budget;
they must not depend on a separately trained model. Retain desired ancestral changes explicitly.
The complete DAG includes every failed/unselected branch and every parent edge. Read details
with paginated tools. Parent recommendations are a mixture of quality, Pareto and novelty;
any fully evaluated valid ancestor can be a parent. This is not UCT/MCTS. No arbitrary shell,
network or dataset access is provided. Custom modules only receive ONE cell's gene token
states [1,L,D]; they cannot see other cells, target labels, batch IDs or dataset objects.
The six reported metrics come from Cajal's five-task evaluation; integration has bio and batch.
Feedback sets are held out from pretraining but reused in research, so results are exploratory,
not a fresh blind test. The sealed V4 test is inaccessible. One shared model must improve the
joint frontier; never combine different architectures' best task results. Valid non-winning
architectures can remain exploration parents. Submit only through submit_candidate after an
accepted CPU synthetic probe. Prose is not a submission. Never invent measurements or ancestry.
No automatic expansion to new data sizes and no upstream recipe adoption is authorized here."""


def loads(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )


def tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool("inspect_contract", "Read candidate schema, tensor API and immutable Cajal protocol."),
    tool(
        "list_components",
        "Read one page of the merged component library and search parameter ranges.",
        {"offset": {"type": "integer", "minimum": 0}, "available_only": {"type": "boolean"}},
        [],
    ),
    tool(
        "baseline_detail",
        "Read current Cajal baseline scores or one existing method's full checkpoint trajectory.",
        {"method": {"type": "string"}},
        [],
    ),
    tool(
        "read_artifact",
        "Read a bounded page of a candidate artifact.",
        {
            "candidate_id": {"type": "string"},
            "kind": {"enum": ["bundle", "validation", "result"]},
            "offset": {"type": "integer", "minimum": 0},
        },
        ["candidate_id", "kind", "offset"],
    ),
    tool(
        "lineage",
        "Read an ancestor's status, metrics and parent IDs.",
        {"candidate_id": {"type": "string"}},
        ["candidate_id"],
    ),
    tool(
        "write_candidate",
        "Stage an immutable candidate. This does not train.",
        {"bundle": candidate_schema()},
        ["bundle"],
    ),
    tool(
        "validate_candidate",
        "Run CPU synthetic five-objective/gradient/reload checks without real data.",
        {"candidate_id": {"type": "string"}},
        ["candidate_id"],
    ),
    tool(
        "submit_candidate",
        "Submit one accepted, nonduplicate candidate for joint evaluation.",
        {"candidate_id": {"type": "string"}},
        ["candidate_id"],
    ),
]


class ToolSession:
    def __init__(self, store, baseline, validator, parents, seen, origin, submission, seen_lock=None):
        self.store, self.baseline, self.validator = store, baseline, validator
        self.parents, self.seen, self.origin = parents, set(seen) if seen_lock is None else seen, origin
        self.seen_lock = seen_lock
        self.submission = submission
        self.staged, self.validated = set(), set()
        self.submitted, self.validation_attempts = None, 0

    def call(self, name, args):
        from .schema import validate

        schema = next((t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == name), None)
        if schema is None:
            raise ValueError("unknown tool")
        validate(args, schema)
        if name == "inspect_contract":
            return {
                "schema": candidate_schema(),
                "metrics": list(METRICS),
                "plugin_kinds": list(KINDS),
                "tensor_api": {
                    "token_transform": "build(hidden_dim, **declared) -> nn.Module; forward(one_cell[1,L,D]) -> same shape",
                    "gene_loss": "compute(prediction[M], target[M], **declared) -> differentiable nonnegative scalar",
                    "classification_loss": "compute(logits[B,C], target[B], **declared); applied to every hierarchy level",
                    "optimizer": "build(parameters, lr, weight_decay, **declared) -> Optimizer covering each parameter once",
                },
                "source_budget": {"per_file_bytes": 32768, "total_bytes": 131072, "full_parameters": 200000000},
                "protocol": "all five joint objectives; fixed 10k checkpoint on 30k schedule; single Cajal seed 0; Cajal unified_1m data",
                "weight_policy": "random initialization; resume only this candidate's own same-seed checkpoint; no historical/pretrained teachers",
            }
        if name == "list_components":
            from .components import catalog

            entries = catalog()
            if args.get("available_only", True):
                entries = [e for e in entries if e["available"] and (e["slots"] or e.get("search_field"))]
            offset = args.get("offset", 0)
            return {
                "total": len(entries),
                "entries": entries[offset : offset + 12],
                "next_offset": offset + 12 if offset + 12 < len(entries) else None,
                "model": MODEL_SPACE,
                "model_ranges": MODEL_RANGES,
                "train_ranges": TRAIN_RANGES,
                "train_integer_ranges": TRAIN_INTS,
                "data_training_sampler_ranges": DATA_RANGES,
                "optimizer_parameter_ranges": OPTIMIZER_RANGES,
                "balancer_parameter_ranges": BALANCER_RANGES,
                "balancer": ["initial", "grad_equal", "famo"],
                "optimizer": ["grouped_muon", "adamw", "adam", "sgd", "custom"],
                "regularizers": ["cond_hsic", "sigreg", "batch_swap", "batch_adversary"],
                "reference": "configs/experiments/cajal_v1.toml",
                "five_objectives_mandatory": True,
            }
        if name == "baseline_detail":
            if args.get("method"):
                from .contracts import ROOT

                methods = json.loads((ROOT / "reference/current_method_scores.json").read_text())
                return {
                    "method": args["method"],
                    "rows": [
                        r
                        for r in methods["rows"]
                        if r["method"] == args["method"] or r["method"].split("@")[0] == args["method"]
                    ],
                    "source": methods["source"],
                    "data_regime": "existing Cajal unified_1m and its held-out evaluation sets",
                }
            return self.baseline
        if name == "read_artifact":
            return self.store.read(args["candidate_id"], args["kind"], args["offset"])
        if name == "lineage":
            identifier = self.store.resolve(args["candidate_id"])
            return {"node": self.store.node(identifier), "parent_ids": self.store.bundle(identifier)["parent_ids"]}
        if name == "write_candidate":
            validate_bundle(args["bundle"], self.parents)
            with self.seen_lock or nullcontext():
                semantic = semantic_id(args["bundle"])
                if semantic in self.seen:
                    raise ValueError("semantic duplicate; metadata/comments are not mutations")
                identifier = self.store.stage(args["bundle"], self.parents, self.origin)
                self.seen.add(semantic)
            self.staged.add(identifier)
            return {"candidate_id": identifier}
        identifier = args["candidate_id"]
        if identifier not in self.staged:
            raise ValueError("validate/submit only a candidate staged in this proposal")
        if name == "validate_candidate":
            self.validation_attempts += 1
            if self.validation_attempts > 4:
                raise ValueError("four acceptance/repair attempts per proposal")
            value = self.validator(identifier, self.store.bundle(identifier))
            if value["status"] == "accepted":
                self.validated.add(identifier)
            return value
        if name == "submit_candidate":
            if identifier not in self.validated or self.submitted:
                raise ValueError("submit exactly one accepted candidate")
            self.submitted = identifier
            save(self.submission, {"candidate_id": identifier, "origin": self.origin})
            return {"status": "submitted", "candidate_id": identifier}
        raise ValueError("unsupported tool")


def propose(client, session, context, max_turns=24, reasoning="max"):
    save(session.submission.with_suffix(".context.json"), context)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(context)}]
    trace, usage = [], {}
    for turn in range(max_turns):
        result = client.chat(
            messages, tools=TOOLS, tool_choice="required", reasoning_effort=reasoning, max_tokens=12000
        )
        for key, value in result.get("usage", {}).items():
            if type(value) is int:
                usage[key] = usage.get(key, 0) + value
        message = result["choices"][0]["message"]
        calls = message.get("tool_calls", [])
        if not calls:
            raise ValueError("ReAct must use tools; prose cannot submit")
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for call in calls:
            try:
                value = session.call(call["function"]["name"], loads(call["function"]["arguments"]))
            except Exception as exc:  # noqa: BLE001 - untrusted tool arguments are returned as repair feedback
                value = {"status": "rejected", "error": type(exc).__name__, "message": str(exc)[:1500]}
            trace.append({"turn": turn, "tool": call["function"]["name"], "response": value})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value, allow_nan=False)})
        save(session.submission.with_suffix(".trace.json"), {"trace": trace, "usage": usage, "messages": messages})
        if session.submitted:
            return {"candidate_id": session.submitted, "usage": usage}
    raise ValueError("ReAct tool-turn budget exhausted")
