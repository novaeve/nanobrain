"""`cajal`: one small command per step, so people and agents compose them instead of writing loops.

    cajal catalog [masks|plans|experiments]       what can be named in a config
    cajal show EXP.toml [--set k=v ...]           the resolved runs of an experiment, validated
    cajal run EXP.toml [--device cpu|cuda] [--gpus 6,7]  launch runs; --dry-run prints the plan
    cajal predict CKPT --data DIR --out X.npz [--device cpu|cuda]    zero-shot outputs of a checkpoint
    cajal eval [--device cpu|cuda] --pred NAME=X.npz ... --out DIR  score against baselines

Every train and eval writes record.json (config, git sha and dirty flag, data manifest hash, versions).
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import experiment, masking

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"


def cmd_catalog(args):
    kinds = [args.kind] if args.kind else ["masks", "losses", "plans", "experiments"]
    regs = []
    if "masks" in kinds:
        regs += [masking.ROW_MASKS, masking.GROUP_MASKS]
    if "losses" in kinds:
        from .balance import BALANCERS
        from .regularizers import OBJECTIVES

        regs += [OBJECTIVES, BALANCERS]
    for reg in regs:
        print(f"[{reg.kind}]")
        for e in reg.describe():
            print(f"  {e['name']:14s} {json.dumps(e['params'], default=str):60s} {e['summary']}")
    for kind, folder in (("plans", "masks"), ("experiments", "experiments")):
        if kind in kinds:
            print(f"[{kind}]  configs/{folder}/")
            for path in sorted((CONFIGS / folder).rglob("*.toml")):
                doc = experiment.load(path)
                name = str(path.relative_to(CONFIGS / folder).with_suffix(""))
                if kind == "experiments" and not name.split("/")[-1].startswith("_"):
                    print(
                        f"  {name:32s} [{doc.get('status', '-'):9s}] {doc.get('outcome') or doc.get('description', '')}"
                    )
                elif kind == "plans":
                    print(f"  {name:32s} {doc.get('description', '')}")


def cmd_show(args):
    runs = experiment.resolve(args.config, args.set)
    print(json.dumps(runs, indent=1, default=str))
    print(f"# {len(runs)} run(s), all valid", file=sys.stderr)


def _train_argv(args, run: dict) -> list[str]:
    sets = [*args.set, *(f"{k}={json.dumps(v)}" for k, v in run.get("run", {}).get("sweep", {}).items())]
    out = run["run"]["out"]
    if "sweep" in run.get("run", {}):  # one output directory per sweep member
        out += "_" + "_".join(str(i) for i in run["run"]["sweep_index"])
    return [
        sys.executable,
        "-m",
        "brainwm_unified.train",
        "--config",
        str(args.config),
        "--out",
        out,
        "--device",
        args.device,
        *(["--set", *sets] if sets else []),
    ]


def cmd_run(args):
    if args.device == "cpu" and args.gpus:
        raise SystemExit("--gpus requires --device cuda")
    runs = experiment.resolve(args.config, args.set)
    sweep = experiment.load(args.config).get("sweep", {})
    for run in runs:  # index of each sweep value, for stable directory names
        if sweep:
            run["run"]["sweep_index"] = [sweep[k].index(run["run"]["sweep"][k]) for k in sorted(sweep)]
    devices = [None] if args.device == "cpu" else ([int(g) for g in args.gpus.split(",")] if args.gpus else [0])
    slots = {device: [] for device in devices}
    for i, run in enumerate(runs):
        slots[devices[i % len(devices)]].append(_train_argv(args, run))
    procs = []
    for gpu, cmds in slots.items():
        if not cmds:
            continue
        chain = " && ".join(shlex.join(c) for c in cmds)  # runs sharing a device go one after another
        label = "CPU" if gpu is None else f"GPU {gpu}"
        print(f"{label}: {chain}")
        if not args.dry_run:
            log = Path(cmds[0][cmds[0].index("--out") + 1] + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            if gpu is None:
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            with open(log, "a") as fh:  # the child keeps its own copy of the descriptor
                procs.append(
                    subprocess.Popen(
                        ["bash", "-c", chain],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=REPO,
                        start_new_session=True,
                    )
                )
            print(f"  started; log {log}")
    if args.wait:  # block until every run has finished (for queues)
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"runs exited with {codes}")


def cmd_predict(args):
    sys.argv = [
        "predict_eval.py", "--checkpoint", args.checkpoint, "--data", args.data,
        "--out", args.out, "--device", args.device,
    ]
    sys.path.insert(0, str(REPO / "scripts"))
    import predict_eval

    predict_eval.main()


def cmd_eval(args):
    script = REPO / "scripts" / "eval_zero_shot.py"
    command = [sys.executable, str(script), "--device", args.device, *args.rest]
    if args.device == "cuda":
        command[0] = "python"
        env_script = REPO / "scripts" / "eval_env.sh"
        command = ["bash", "-c", 'source "$1" && shift && exec "$@"', "cajal-eval", str(env_script), *command]
    raise SystemExit(subprocess.call(command, cwd=REPO))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cajal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("catalog", help="registered components, mask plans and experiments")
    p.add_argument("kind", nargs="?", choices=["masks", "losses", "plans", "experiments"])
    p.set_defaults(fn=cmd_catalog)
    for name, fn, helptext in (
        ("show", cmd_show, "resolve and validate an experiment"),
        ("run", cmd_run, "launch an experiment's runs"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("config", type=Path)
        p.add_argument("--set", nargs="*", action="extend", default=[], help="overrides, e.g. train.steps=1000")
        if name == "run":
            p.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="training device (default: cuda)")
            p.add_argument("--gpus", help="comma-separated GPU ids; runs are spread over them")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--wait", action="store_true", help="return only when all runs have finished")
        p.set_defaults(fn=fn)
    p = sub.add_parser("predict", help="zero-shot outputs of a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--data", default="/data/brainwm/eval_10k")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("eval", help="score predictions (arguments go to scripts/eval_zero_shot.py)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.set_defaults(fn=cmd_eval)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "eval":
        args.rest = rest[1:] if rest[:1] == ["--"] else rest
    elif rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    args.fn(args)


if __name__ == "__main__":
    main()
