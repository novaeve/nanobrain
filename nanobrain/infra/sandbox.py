"""Credential-free Linux workers. Probe never mounts benchmark data or GPUs."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def command(root, output, arguments, *, inputs=(), gpu=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).absolute()
    info = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-B",
                "-c",
                "import sys,json;print(json.dumps(dict(prefix=sys.base_prefix,venv=sys.prefix,paths=sys.path)))",
            ],
            text=True,
        )
    )
    mounts = {Path(info["prefix"]).resolve(), Path(info["venv"]).resolve()}
    mounts |= {Path(p).resolve() for p in info["paths"] if p and "site-packages" in p and Path(p).is_dir()}
    mounts |= {
        Path(p)
        for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/ld.so.cache", "/etc/alternatives")
        if Path(p).exists()
    }
    args = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--hostname",
        "cajal-worker",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    bound = []
    for p in sorted(mounts, key=lambda p: len(str(p))):
        if not any(p.is_relative_to(q) for q in bound):
            args += ["--ro-bind", str(p), str(p)]
            bound.append(p)
    # Invoke the same venv even when .venv is a symlink shared with the upstream checkout.
    alias = python.parent.parent
    if not any(alias.is_relative_to(q) for q in bound):
        args += ["--ro-bind", info["venv"], str(alias)]
    source_entries = {
        "brainwm_unified",
        "nanobrain",
        "scripts",
        "configs",
        "reference",
        ".git",
        "toolchain",
        ".component_deps",
    }
    if (root / ".git").exists():
        tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split("\0")
        # Every tracked file must exist for Cajal's own git_dirty provenance check.
        # Only tracked source roots are exposed; .env, preparation and research-runs stay hidden.
        source_entries.update(Path(name).parts[0] for name in tracked if name)
    for rel in sorted(source_entries):
        p = root / rel
        if p.exists():
            args += ["--ro-bind", str(p.resolve()), str(p)]
    for p in inputs:
        p = Path(p).absolute()
        if not p.exists():
            raise FileNotFoundError(p)
        args += ["--ro-bind", str(p.resolve()), str(p)]
    env = {
        "PATH": "/usr/local/cuda-13.0/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "USER": "cajal-worker",
        "LOGNAME": "cajal-worker",
        "TMPDIR": "/tmp",
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "",
        "CUDA_HOME": "/usr/local/cuda-13.0",
        "TORCHINDUCTOR_CACHE_DIR": "/work/inductor-cache",
        "NUMBA_CACHE_DIR": "/tmp/numba",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "LD_LIBRARY_PATH": str(root / "toolchain/lib")
        + ":"
        + ":".join(
            str(Path(info["venv"]) / "lib/python3.11/site-packages/nvidia" / p / "lib")
            for p in ("nvjitlink", "cusparse")
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if gpu is not None:
        if type(gpu) is not int or gpu not in range(5):
            raise ValueError("GPU whitelist is 0..4; GPUs 5, 6 and 7 are excluded by the user")
        for name in ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", f"nvidia{gpu}"):
            p = Path("/dev") / name
            if p.exists():
                args += ["--dev-bind", str(p), str(p)]
        args += ["--ro-bind", "/sys", "/sys"]
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if not uuid.startswith("GPU-") or "\n" in uuid:
            raise ValueError("invalid GPU identity")
        env["CUDA_VISIBLE_DEVICES"] = uuid
    args += ["--bind", str(output), "/work", "--chdir", "/work"]
    for key, value in env.items():
        args += ["--setenv", key, value]
    return args + ["--", str(python), "-B", "-m", "nanobrain.infra.worker", *arguments]


def clean_environment():
    # This server's bwrap predates --clearenv; Popen receives only this whitelist.
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def launch(root, output, arguments, *, inputs=(), gpu=None, timeout=600, on_started=None):
    output = Path(output)
    args = command(root, output, arguments, inputs=inputs, gpu=gpu)
    with (output / "worker.log").open("w") as log:
        p = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=clean_environment()
        )
        try:
            if on_started is not None:
                on_started(p.pid)
            code = p.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    if code:
        raise RuntimeError(f"sandbox worker exited {code}: {output / 'worker.log'}")
    return json.loads((output / "receipt.json").read_text())
