"""
scUniFlow flat-order training script.

Identical to train.py except:
- No spectral coordinate file required
- Gene-to-slot routing is sequential: gene i -> slot i
  (SpectralRouter default: slot_to_gene[:G] = arange(G))
- gene_embedding uses default random init (Normal(0, 0.02))
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import yaml
import torch
import torch.distributed as dist

from data import load_sea_ad, load_sea_ad_split, scRNADataset
from paths import ensure_tmp_root
from train import (
    DistributedEvalSampler,
    cleanup_ddp,
    compute_log1p_nonzero_prior_stats,
    is_main,
    make_dataloader,
    resolve_epoch_training_config,
    scRNAPipelineTrainer,
    setup_ddp,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scUniFlow 顺序展平路由训练（无谱嵌入）")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data", type=str, default="mtg_toy",
                        help="数据键名或 h5ad 路径")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="单卡 batch size，覆盖 config 中的 batch_size_per_gpu")
    parser.add_argument("--phase", type=str, default=None,
                        help="已弃用；当前仅保留兼容，支持 all/single 或不传")
    parser.add_argument("--run_name", type=str, default=None,
                        help="实验名称，默认为 flat_{data}_{timestamp}")
    parser.add_argument("--max_cells", type=int, default=None,
                        help="仅抽样部分细胞做快速验证；对 train/val 总数生效")
    parser.add_argument("--debug_t_bucket_stats", action="store_true",
                        help="打印训练时按 t 分桶的 x1_pred/x1_true 均值统计")
    parser.add_argument("--debug_t_bucket_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 路径恢复训练（.pt 文件）")
    args = parser.parse_args()

    if args.phase not in (None, "all", "single"):
        raise ValueError("当前训练已重构为单阶段 epoch 模式，不再支持 --phase 1/2。请移除该参数或使用 --phase all。")

    rank, local_rank, world_size = setup_ddp()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    run_name = args.run_name or f"flat_{args.data}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(train_cfg.get("experiment_dir", "experiments"), run_name)
    if is_main():
        os.makedirs(exp_dir, exist_ok=True)
        # Save config with routing mode annotation
        cfg_out = dict(cfg)
        cfg_out["routing"] = {"mode": "flat_sequential", "spectral_coords": None}
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg_out, f, default_flow_style=False)

    if world_size > 1:
        dist.barrier()

    batch_size = args.batch_size or int(train_cfg.get("batch_size_per_gpu", 32))
    val_cfg_early = cfg.get("validation", {})
    val_batch_size = int(val_cfg_early.get("batch_size", batch_size))

    train_nonzero_mask_ratio = float(train_cfg.get("nonzero_mask_ratio", 0.2))
    train_zero_mask_ratio = float(train_cfg.get("zero_mask_ratio", 0.0))
    seed = train_cfg.get("seed", 42)
    max_cells = args.max_cells if args.max_cells is not None else train_cfg.get("max_cells")
    cell_seed = int(train_cfg.get("cell_seed", seed))

    val_ratio = train_cfg.get("val_ratio", 0.1)

    if is_main():
        print("=" * 60)
        print(f"实验目录: {exp_dir}")
        print(f"路由模式: 顺序展平 (gene i -> slot i)")
        print(f"GPU 数量: {world_size}")
        print("=" * 60)

    # ---- 数据加载 ----
    use_split = args.data in ("mtg_full", "dlpfc_full") or val_ratio > 0
    val_loader = None
    train_sampler = None

    if rank == 0:
        if use_split:
            try:
                train_data, val_data, _, meta = load_sea_ad_split(
                    args.data,
                    val_ratio=val_ratio,
                    seed=seed,
                    max_cells=max_cells,
                    cell_seed=cell_seed,
                )
                train_X, gene_ids = train_data
                val_X, _ = val_data
                n_genes = meta["n_genes"]
                train_prior_mean, train_prior_std = compute_log1p_nonzero_prior_stats(train_X)

                split_path = os.path.join(exp_dir, "split_info.json")
                with open(split_path, "w") as f:
                    json.dump({
                        "train_donors": meta["train_donors"],
                        "val_donors": meta["val_donors"],
                        **meta["split_info_effective"],
                    }, f, indent=2, ensure_ascii=False)
                print(f"[saved] 划分信息 -> {split_path}")

                if world_size > 1:
                    import tempfile, pickle
                    shard_dir = tempfile.mkdtemp(prefix="scmam_flat_shards_", dir=ensure_tmp_root())
                    n_train = train_X.shape[0]
                    chunk = n_train // world_size
                    for r in range(world_size):
                        start = r * chunk
                        end = n_train if r == world_size - 1 else (r + 1) * chunk
                        shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                        with open(shard_path, "wb") as sf:
                            pickle.dump(train_X[start:end], sf)
                    info_path = os.path.join(shard_dir, "info.pkl")
                    with open(info_path, "wb") as sf:
                        pickle.dump({
                            "gene_ids": gene_ids, "n_genes": n_genes,
                            "val_X": val_X, "shard_dir": shard_dir,
                            "train_prior_mean": train_prior_mean,
                            "train_prior_std": train_prior_std,
                        }, sf)
                    shard_dir_bytes = shard_dir.encode()
                else:
                    shard_dir_bytes = b""
            except (FileNotFoundError, KeyError):
                print("[warn] 无法按 donor 划分，回退到无验证集模式")
                count_matrix, gene_ids, _, meta = load_sea_ad(
                    args.data,
                    max_cells=max_cells,
                    cell_seed=cell_seed,
                )
                n_genes = meta["n_genes"]
                train_prior_mean, train_prior_std = compute_log1p_nonzero_prior_stats(count_matrix)
                use_split = False
                if world_size > 1:
                    import tempfile, pickle
                    shard_dir = tempfile.mkdtemp(prefix="scmam_flat_shards_", dir=ensure_tmp_root())
                    n_total = count_matrix.shape[0]
                    chunk = n_total // world_size
                    for r in range(world_size):
                        start = r * chunk
                        end = n_total if r == world_size - 1 else (r + 1) * chunk
                        shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                        with open(shard_path, "wb") as sf:
                            pickle.dump(count_matrix[start:end], sf)
                    info_path = os.path.join(shard_dir, "info.pkl")
                    with open(info_path, "wb") as sf:
                        pickle.dump({
                            "gene_ids": gene_ids, "n_genes": n_genes,
                            "val_X": None, "shard_dir": shard_dir,
                            "train_prior_mean": train_prior_mean,
                            "train_prior_std": train_prior_std,
                        }, sf)
                    shard_dir_bytes = shard_dir.encode()
                else:
                    shard_dir_bytes = b""
        else:
            count_matrix, gene_ids, _, meta = load_sea_ad(
                args.data,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
            n_genes = meta["n_genes"]
            train_prior_mean, train_prior_std = compute_log1p_nonzero_prior_stats(count_matrix)
            if world_size > 1:
                import tempfile, pickle
                shard_dir = tempfile.mkdtemp(prefix="scmam_flat_shards_", dir=ensure_tmp_root())
                n_total = count_matrix.shape[0]
                chunk = n_total // world_size
                for r in range(world_size):
                    start = r * chunk
                    end = n_total if r == world_size - 1 else (r + 1) * chunk
                    shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                    with open(shard_path, "wb") as sf:
                        pickle.dump(count_matrix[start:end], sf)
                info_path = os.path.join(shard_dir, "info.pkl")
                with open(info_path, "wb") as sf:
                    pickle.dump({
                        "gene_ids": gene_ids, "n_genes": n_genes,
                        "val_X": None, "shard_dir": shard_dir,
                        "train_prior_mean": train_prior_mean,
                        "train_prior_std": train_prior_std,
                    }, sf)
                shard_dir_bytes = shard_dir.encode()
            else:
                shard_dir_bytes = b""

    if world_size > 1:
        import pickle
        obj_list = [shard_dir_bytes if rank == 0 else b""]
        dist.broadcast_object_list(obj_list, src=0)
        shard_dir = obj_list[0].decode()

        info_path = os.path.join(shard_dir, "info.pkl")
        with open(info_path, "rb") as sf:
            info = pickle.load(sf)
        gene_ids = info["gene_ids"]
        n_genes = info["n_genes"]
        train_prior_mean = float(info.get("train_prior_mean", 0.0))
        train_prior_std = float(max(info.get("train_prior_std", 1.0), 1e-6))

        my_shard_path = os.path.join(shard_dir, f"train_{rank}.pkl")
        with open(my_shard_path, "rb") as sf:
            my_train_X = pickle.load(sf)
        print(f"[rank {rank}] 加载分片: {my_train_X.shape[0]:,} cells")

        train_ds = scRNADataset(
            my_train_X,
            gene_ids,
            x0_prior="standard_normal",
            nonzero_mask_ratio=train_nonzero_mask_ratio,
            zero_mask_ratio=train_zero_mask_ratio,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )

        val_X = info["val_X"]
        if val_X is not None:
            val_ds = scRNADataset(
                val_X,
                gene_ids,
                x0_prior="standard_normal",
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
            )
            val_sampler = DistributedEvalSampler(val_ds) if world_size > 1 else None
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=val_batch_size, shuffle=False,
                sampler=val_sampler, num_workers=4, pin_memory=True,
            )

        del my_train_X
        dist.barrier()
    else:
        if use_split:
            train_loader, train_sampler = make_dataloader(
                train_X, gene_ids,
                batch_size=batch_size, shuffle=True, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
            )
            val_loader, _ = make_dataloader(
                val_X, gene_ids,
                batch_size=val_batch_size, shuffle=False, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
            )
        else:
            train_loader, train_sampler = make_dataloader(
                count_matrix, gene_ids,
                batch_size=batch_size, shuffle=True, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
            )

    # ---- 模型 ----
    if is_main():
        print("=" * 60)
        print(f"模型: n_genes={n_genes}, d_model={model_cfg['d_model']}, "
              f"grid_size={model_cfg['grid_size']}")
        print(f"路由: 顺序展平 gene[0..{n_genes-1}] -> slot[0..{n_genes-1}]")
        print("=" * 60)

    trainer = scRNAPipelineTrainer(
        n_genes=n_genes,
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
        local_rank=local_rank,
        exp_dir=exp_dir,
    )
    loss_cfg = train_cfg.get("loss_weights", {})
    trainer.loss_weighted_x1 = float(loss_cfg.get("weighted_x1", 1.0))
    trainer.loss_pearson_weight = float(loss_cfg.get("pearson", 0.1))
    trainer.loss_cell_pearson_weight = float(loss_cfg.get("cell_pearson", 0.1))
    trainer.debug_t_bucket_stats = bool(args.debug_t_bucket_stats)
    trainer.debug_t_bucket_every = max(1, int(args.debug_t_bucket_every))
    val_cfg = cfg.get("validation", {})
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", cfg.get("inference", {}).get("temperature", 0.01)))
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.0))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    val_max_batches = int(val_cfg.get("max_batches", 200))
    trainer.val_max_batches = val_max_batches
    trainer.val_prior_mean = float(train_prior_mean)
    trainer.val_prior_std = float(max(train_prior_std, 1e-6))

    if is_main():
        print(
            f"[val prior] mean={trainer.val_prior_mean:.6f}, std={trainer.val_prior_std:.6f}"
        )
        # Verify default sequential routing is in place
        raw = trainer.raw_model
        s2g = raw.router.slot_to_gene.cpu().numpy()
        g2s = raw.router.gene_to_slot.cpu().numpy()
        assert (s2g[:n_genes] == np.arange(n_genes)).all(), "Default routing is not sequential"
        assert (g2s == np.arange(n_genes)).all(), "Default routing is not sequential"
        print(f"[init] 顺序展平路由已确认: gene[i] -> slot[i], 无谱嵌入")

    if world_size > 1:
        dist.barrier()

    # ---- 训练 ----
    train_schedule = resolve_epoch_training_config(train_cfg, steps_per_epoch=len(train_loader))
    if is_main():
        print(
            "[train schedule] "
            f"epochs={train_schedule['epochs']}, "
            f"steps/epoch={len(train_loader)}, "
            f"lr={train_schedule['lr']:.2e}, "
            f"min_lr={train_schedule['min_lr']:.2e}, "
            f"val_every_epochs={train_schedule['val_every_epochs']}, "
            f"save_every_epochs={train_schedule['save_every_epochs']}"
        )

    trainer.train_epochs(
        train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        epochs=train_schedule["epochs"],
        lr=train_schedule["lr"],
        min_lr=train_schedule["min_lr"],
        val_every_epochs=train_schedule["val_every_epochs"],
        save_every_epochs=train_schedule["save_every_epochs"],
        resume_ckpt=args.resume,
    )

    # 最终模型保存
    if is_main():
        final_path = os.path.join(exp_dir, "model_final.pt")
        torch.save(trainer.raw_model.state_dict(), final_path)
        print(f"\n最终模型 -> {final_path}")

    cleanup_ddp()
