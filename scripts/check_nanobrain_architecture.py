#!/usr/bin/env python3
"""Validate/inspect a graph genome or export a legacy genome, without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nanobrain.genome import CandidateGenome
from nanobrain.architecture import plan_graph, Shape, graph_catalog
from nanobrain.architecture_migration import legacy_to_graph
from nanobrain.evolution import architecture_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", type=Path)
    parser.add_argument("--migrate-to", type=Path, help="write a NEW graph genome; never overwrite")
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--report", type=Path, help="write a NEW validation report")
    args = parser.parse_args()
    payload = json.loads(args.genome.read_text())
    genome = CandidateGenome.from_dict(payload)
    if args.migrate_to:
        spec = legacy_to_graph(genome, args.input_dim)
        genome = CandidateGenome.from_dict(dict(architecture=spec, learning_rate=genome.learning_rate, seed=genome.seed))
        with args.migrate_to.open("x") as f:
            json.dump(genome.to_dict(), f, indent=2); f.write("\n")
    if genome.architecture is None:
        parser.error("legacy genome: use --migrate-to NEW_PATH first")
    spec = genome.to_dict()["architecture"]
    report = dict(architecture_id=architecture_id(genome.to_dict()), version=2, status="valid", graphs={})
    for key, inputs in (("backbone", {"features": Shape(spec["input_dim"])}),
                        ("flow", {"state": Shape(spec["width"]), "context": Shape(spec["width"]), "time": Shape(3)})):
        plan, output, used = plan_graph(spec[key], inputs, allow_spatial=key == "backbone")
        report["graphs"][key] = dict(used_inputs=sorted(used), output=vars(output),
            nodes=[{**n, "shape": vars(n["shape"])} for n in plan])
    if args.report:
        with args.report.open("x") as f:
            json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
