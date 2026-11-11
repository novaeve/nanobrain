import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from model import scFiTModel
from utils import seed_everything

from .protocol import load_protocol_bundle
from .train import RouteATrainer


def _configure_cpu_thread_env():
    defaults = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_configure_cpu_thread_env()

try:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _env_flag_is_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _assert_required_cuda_available():
    if not _env_flag_is_true("SCFIT_REQUIRE_CUDA", default=False):
        return
    if torch.cuda.is_available():
        return

    raise RuntimeError(
        "SCFIT_REQUIRE_CUDA=1 but torch.cuda.is_available() is False. "
        f"python={sys.executable} "
        f"torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}. "
        "This usually means the installed PyTorch CUDA build is incompatible with the host NVIDIA driver, "
        "or the CUDA runtime libraries in the current environment are misconfigured."
    )


def build_experiment_dir(root: str, run_name: str) -> str:
    exp_dir = os.path.join(root, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    return exp_dir


def _archive_existing_train_log_if_fresh_run(exp_dir: str, resume: str = None):
    if resume:
        return
    log_path = Path(exp_dir) / "train_log.jsonl"
    if not log_path.exists() or log_path.stat().st_size == 0:
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    archive_path = log_path.with_name(f"train_log.before_fresh_run.{ts}.jsonl")
    log_path.rename(archive_path)
    print(f"[train_log] archived existing log -> {archive_path}")


class GpuMemoryReservation:
    def __init__(self, device: torch.device, enabled: bool = False, reserve_gb: float = None, fraction_of_free: float = 0.8):
        self.device = device
        self.enabled = bool(enabled) and device.type == "cuda"
        self.reserve_gb = None if reserve_gb is None else float(reserve_gb)
        self.fraction_of_free = float(fraction_of_free)
        self.buffer = None

    def reserve(self):
        if not self.enabled:
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        if self.reserve_gb is not None:
            target_bytes = int(self.reserve_gb * (1024 ** 3))
        else:
            safe_fraction = min(max(self.fraction_of_free, 0.0), 0.95)
            target_bytes = int(free_bytes * safe_fraction)

        reserve_cap = int(free_bytes * 0.9)
        target_bytes = max(0, min(target_bytes, reserve_cap))
        if target_bytes <= 0:
            print("[gpu_reserve] skip: no free memory available to reserve")
            return

        try:
            self.buffer = torch.empty(target_bytes, dtype=torch.uint8, device=self.device)
            self.buffer.zero_()
            print(
                "[gpu_reserve] "
                f"reserved={target_bytes / (1024 ** 3):.2f}GB "
                f"free_before={free_bytes / (1024 ** 3):.2f}GB "
                f"total={total_bytes / (1024 ** 3):.2f}GB"
            )
        except RuntimeError as exc:
            self.buffer = None
            print(f"[gpu_reserve] failed: {exc}")

    def release(self):
        if self.buffer is None:
            return
        self.buffer = None
        torch.cuda.empty_cache()
        print("[gpu_reserve] released reservation")


def run_route_a_training(cfg: dict):
    protocol_cfg = cfg.get("protocol", {}) or {}
    protocol_path = str(protocol_cfg.get("path", "") or "").strip()
    if not protocol_path:
        raise ValueError("Route A config requires `protocol.path`")

    train_cfg = cfg["training"]
    exp_cfg = cfg["experiment"]
    reserve_cfg = train_cfg.get("gpu_reservation", {})
    seed_everything(int(train_cfg.get("seed", 42)))

    output_root = str(exp_cfg.get("output_dir", "") or "").strip()
    if not output_root or output_root.startswith("<"):
        raise ValueError("Please set `experiment.output_dir` in the config to a writable output directory.")
    run_name = exp_cfg.get("run_name") or Path(protocol_path).stem
    exp_dir = build_experiment_dir(output_root, run_name)
    _archive_existing_train_log_if_fresh_run(exp_dir, resume=train_cfg.get("resume"))
    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    _assert_required_cuda_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_reservation = GpuMemoryReservation(
        device=device,
        enabled=bool(reserve_cfg.get("enabled", False)),
        reserve_gb=reserve_cfg.get("reserve_gb"),
        fraction_of_free=float(reserve_cfg.get("fraction_of_free", 0.8)),
    )
    gpu_reservation.reserve()

    bundle = load_protocol_bundle(protocol_path)
    model = scFiTModel(
        n_genes=int(bundle["bundle_meta"]["n_genes"]),
        top_k_genes=int(bundle["bundle_meta"]["fixed_panel_size"]),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        d_ff=int(cfg["model"]["d_ff"]),
        nlayers=int(cfg["model"]["nlayers"]),
        dropout=float(cfg["model"]["dropout"]),
        value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
        time_conditioning_mode=str(cfg["model"].get("time_conditioning", {}).get("mode", "adaln")),
        aux_bin_ce=False,
        n_bins=0,
        n_cell_types=0,
        pseudotime_head=False,
    )
    gpu_reservation.release()

    trainer = RouteATrainer(model=model, device=device, exp_dir=exp_dir, cfg=cfg, bundle=bundle)
    print("=" * 60)
    print(f"exp_dir={exp_dir}")
    print(f"device={device}")
    print(f"protocol={bundle['protocol_name']}")
    print(f"observed_bins={len(bundle['observed_time_bins'])} heldout_bins={len(bundle['heldout_time_bins'])}")
    print(f"model_input_dim={bundle['bundle_meta']['fixed_panel_size']} total_gene_space={bundle['bundle_meta']['n_genes']}")
    print("=" * 60)
    trainer.train(
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        save_every_epochs=int(train_cfg["save_every_epochs"]),
        resume=train_cfg.get("resume"),
    )


def main():
    parser = argparse.ArgumentParser(description="Train scFiT on Route A")
    parser.add_argument("--config", required=True, help="Route A YAML config path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    protocol_path = str((cfg.get("protocol", {}) or {}).get("path", "") or "").strip()
    if not protocol_path:
        raise ValueError(
            "This entrypoint requires `protocol.path` in the config."
        )
    run_route_a_training(cfg)


if __name__ == "__main__":
    main()
