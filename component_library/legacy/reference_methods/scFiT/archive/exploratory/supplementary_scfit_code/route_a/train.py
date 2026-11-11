import json
import os
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import (
    MASK_CLS,
    MASK_KNOWN,
    build_mask_states,
    pairwise_mean_sq_dist,
    sample_t_stratified,
    sample_transport_pairs,
    sinkhorn_distance,
    sinkhorn_uniform_transport,
)
from .assets import prepare_route_a_assets
from .eval import evaluate_artifact_dir, save_prediction_artifact
from .protocol import load_protocol_bundle, resolve_generation_sources


class RouteATrainer:
    def __init__(self, model, device, exp_dir: str, cfg: Dict, bundle: Dict):
        self.model = model.to(device)
        self.device = device
        self.exp_dir = exp_dir
        self.cfg = cfg
        self.bundle = bundle
        self.log_path = os.path.join(exp_dir, "train_log.jsonl")

        train_cfg = cfg["training"]
        val_cfg = cfg.get("validation", {}) or {}
        eval_cfg = cfg.get("evaluation", {}) or {}
        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.eval_batch_size = int(eval_cfg.get("batch_size", self.batch_size))
        self.infer_micro_batch_size = int(
            cfg.get("inference", {}).get("micro_batch_size", max(1, min(self.eval_batch_size, self.batch_size, 4)))
        )
        self.steps_per_epoch = int(train_cfg.get("iters_per_epoch", train_cfg.get("steps_per_epoch", 200)))
        self.log_every_steps = int(train_cfg.get("log_every_steps", 20))
        self.val_every_steps = int(train_cfg.get("val_every_steps", 500))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.time_pair_sampling = str(train_cfg.get("time_pair_sampling", "uniform"))
        self.time_pair_distance_power = float(train_cfg.get("time_pair_distance_power", 1.0))
        self.time_pair_min_weight = float(train_cfg.get("time_pair_min_weight", 1e-3))
        self.transport_pairing = str(train_cfg.get("transport_pairing", "ot")).strip().lower()
        self.prediction_parameterization = str(
            train_cfg.get("prediction_parameterization", "direct_velocity")
        ).strip().lower()
        self.infer_n_steps = int(cfg.get("inference", {}).get("n_steps", 10))
        self.validation_max_cells_per_time = int(val_cfg.get("max_cells_per_time", 256))
        self.final_eval_max_cells_per_time = int(eval_cfg.get("max_cells_per_time", 0))
        raw_val_metric_names = val_cfg.get("metric_names", ["wasserstein_pca50"])
        if isinstance(raw_val_metric_names, str):
            raw_val_metric_names = [raw_val_metric_names]
        self.validation_metric_names = [str(x).strip() for x in raw_val_metric_names if str(x).strip()]
        if not self.validation_metric_names:
            self.validation_metric_names = ["wasserstein_pca50"]

        ot_cfg = cfg.get("ot", {}) or {}
        self.ot_epsilon = float(ot_cfg.get("epsilon", 0.05))
        self.ot_max_iter = int(ot_cfg.get("max_iter", 100))

        loss_cfg = train_cfg.get("loss_weights", {}) or {}
        self.loss_cfm = float(loss_cfg.get("cfm", 1.0))
        self.loss_wass_hvg = float(loss_cfg.get("wasserstein_hvg", 0.5))
        self.loss_wass_pca = float(loss_cfg.get("wasserstein_pca50", 0.1))
        self.use_wass_pca_loss = self.loss_wass_pca > 0.0
        self.loss_mean = float(loss_cfg.get("mean", 0.1))
        self.loss_variance = float(loss_cfg.get("variance", 0.0))
        self.variance_use_log = bool(train_cfg.get("variance_loss_use_log", True))
        self.variance_eps = float(train_cfg.get("variance_loss_eps", 1e-4))
        self.flow_eps = float(train_cfg.get("flow_eps", 1e-3))

        self.sequence_len = int(self.bundle["bundle_meta"]["fixed_panel_size"]) + 1
        self.total_gene_space = int(self.bundle["bundle_meta"]["n_genes"])
        self.cls_token_id = int(self.total_gene_space)
        hvg_gene_ids = np.asarray(self.bundle["bundle_meta"]["hvg_gene_indices"], dtype=np.int64)
        self.gene_ids_np = np.concatenate([[self.cls_token_id], hvg_gene_ids], axis=0).astype(np.int64)
        self.valid_mask_np = np.ones((self.sequence_len,), dtype=np.bool_)
        self.mask_states_np = np.full((self.sequence_len,), MASK_KNOWN, dtype=np.int64)
        self.mask_states_np[0] = MASK_CLS

        assets_dir = str(cfg.get("evaluation", {}).get("assets_dir", "") or "").strip()
        if not assets_dir or assets_dir.startswith("<"):
            raise ValueError("Please set `evaluation.assets_dir` in the config to a writable evaluation-assets directory.")
        self.assets_dir = assets_dir
        summary_path = Path(self.assets_dir) / "asset_summary.json"
        if not summary_path.exists():
            print(f"[route_a_train] missing assets, preparing now -> {self.assets_dir}")
            prepare_route_a_assets(self.bundle["protocol_path"], self.assets_dir)
        self._load_assets()
        self.training_pairs, self.training_pair_weights = self._build_training_pairs()
        if self.transport_pairing not in {"ot", "random_uniform"}:
            raise ValueError(f"unsupported transport_pairing={self.transport_pairing}")
        if self.prediction_parameterization not in {"direct_velocity", "target_reparam"}:
            raise ValueError(f"unsupported prediction_parameterization={self.prediction_parameterization}")

    def _load_assets(self):
        summary = json.loads((Path(self.assets_dir) / "asset_summary.json").read_text())
        pca = joblib.load(summary["pca_model_path"])
        self.pca_mean = torch.as_tensor(pca.mean_, dtype=torch.float32, device=self.device)
        self.pca_components = torch.as_tensor(pca.components_, dtype=torch.float32, device=self.device)

    def _build_training_pairs(self):
        observed_bins = self.bundle["observed_time_bins"]
        direction_mode = self.bundle["direction_mode"]
        pairs = []
        weights = []
        for src in observed_bins:
            src_t = float(self.bundle["metadata_df"].iloc[self.bundle["time_to_indices"][src][0]]["author_day_numeric"])
            for tgt in observed_bins:
                if src == tgt:
                    continue
                tgt_t = float(self.bundle["metadata_df"].iloc[self.bundle["time_to_indices"][tgt][0]]["author_day_numeric"])
                if direction_mode == "forward_only" and not (src_t < tgt_t):
                    continue
                delta = abs(float(tgt_t) - float(src_t))
                forward = float(tgt_t) > float(src_t)
                pairs.append((src, tgt))
                weights.append(self._time_pair_weight(delta=delta, forward=forward))
        if not pairs:
            raise ValueError("no observed-observed training pairs could be built")
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("invalid training pair weights")
        weights = weights / weights.sum()
        return pairs, weights

    def _time_pair_weight(self, delta: float, forward: bool) -> float:
        delta = max(float(delta), 1e-8)
        mode = self.time_pair_sampling
        power = float(self.time_pair_distance_power)
        min_w = float(self.time_pair_min_weight)
        if mode == "uniform":
            return 1.0
        if mode == "distance_weighted":
            return max(delta**power, min_w)
        if mode == "inverse_distance":
            return max((1.0 / delta) ** max(power, 1e-8), min_w)
        if mode == "forward_only":
            return 1.0 if forward else min_w
        if mode == "forward_distance_weighted":
            return max((delta**power) if forward else min_w, min_w)
        raise ValueError(f"unsupported time_pair_sampling={mode}")

    def _sample_training_pair(self):
        idx = int(np.random.choice(len(self.training_pairs), p=self.training_pair_weights))
        return self.training_pairs[idx]

    def _log(self, payload: Dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, epoch: int, global_step: int, optimizer, tag: str = None):
        ckpt_path = Path(self.exp_dir) / "checkpoints" / f"{tag or f'epoch_{epoch}'}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "protocol_path": self.bundle["protocol_path"],
            },
            ckpt_path,
        )
        print(f"[checkpoint] {ckpt_path}")

    def _sample_cells(self, time_bin: str, n_cells: int):
        idx = self.bundle["time_to_indices"][time_bin]
        choice = np.random.choice(idx, size=int(n_cells), replace=(idx.shape[0] < int(n_cells)))
        expr = np.asarray(self.bundle["expression"][choice], dtype=np.float32)
        return torch.as_tensor(expr, dtype=torch.float32, device=self.device), choice.astype(np.int64)

    def _seq_from_expr(self, expr: torch.Tensor):
        batch_size = expr.shape[0]
        seq = torch.zeros((batch_size, self.sequence_len), dtype=torch.float32, device=self.device)
        seq[:, 1:] = expr
        gene_ids = torch.as_tensor(self.gene_ids_np, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = torch.as_tensor(self.valid_mask_np, dtype=torch.bool, device=self.device).unsqueeze(0).expand(batch_size, -1)
        mask_states = torch.as_tensor(self.mask_states_np, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        padding_mask = ~valid_mask
        return gene_ids, seq, mask_states, padding_mask

    def _pca_transform_torch(self, expr: torch.Tensor):
        return torch.matmul(expr - self.pca_mean.unsqueeze(0), self.pca_components.t())

    def _variance_regularization_loss(self, pred_expr: torch.Tensor, tgt_expr: torch.Tensor):
        if self.loss_variance <= 0.0:
            return pred_expr.new_tensor(0.0)
        if pred_expr.shape[0] < 2 or tgt_expr.shape[0] < 2:
            return pred_expr.new_tensor(0.0)

        pred_var = torch.var(pred_expr, dim=0, unbiased=False)
        tgt_var = torch.var(tgt_expr, dim=0, unbiased=False)
        if self.variance_use_log:
            pred_stat = torch.log(pred_var + self.variance_eps)
            tgt_stat = torch.log(tgt_var + self.variance_eps)
        else:
            pred_stat = pred_var
            tgt_stat = tgt_var
        return torch.mean((pred_stat - tgt_stat) ** 2)

    def _forward_raw_prediction(self, expr: torch.Tensor, t_flow: torch.Tensor, t_src: float, t_tgt: float):
        gene_ids, seq, mask_states, padding_mask = self._seq_from_expr(expr)
        t_src_tensor = torch.full((expr.shape[0], 1), float(t_src), dtype=torch.float32, device=self.device)
        t_tgt_tensor = torch.full((expr.shape[0], 1), float(t_tgt), dtype=torch.float32, device=self.device)
        pred = self.model.forward(
            gene_ids,
            seq,
            mask_states,
            padding_mask,
            t_flow,
            t_src=t_src_tensor,
            t_tgt=t_tgt_tensor,
            cell_type_ids=None,
        )
        return pred[:, 1:]

    def _predict_target_and_velocity(self, expr: torch.Tensor, t_flow: torch.Tensor, t_src: float, t_tgt: float):
        raw_pred = self._forward_raw_prediction(expr, t_flow=t_flow, t_src=t_src, t_tgt=t_tgt)
        denom = (1.0 - t_flow).clamp_min(self.flow_eps)
        if self.prediction_parameterization == "direct_velocity":
            pred_vel = raw_pred
            pred_tgt = torch.clamp(expr + denom * pred_vel, min=0.0)
            return pred_tgt, pred_vel
        if self.prediction_parameterization == "target_reparam":
            pred_tgt = torch.clamp(raw_pred, min=0.0)
            pred_vel = (pred_tgt - expr) / denom
            return pred_tgt, pred_vel
        raise ValueError(f"unsupported prediction_parameterization={self.prediction_parameterization}")

    def _rollout(self, src_expr: np.ndarray, t_src: float, t_tgt: float, n_steps: int):
        outputs = []
        micro_bs = max(1, int(self.infer_micro_batch_size))
        for start in range(0, src_expr.shape[0], micro_bs):
            end = min(start + micro_bs, src_expr.shape[0])
            x = torch.as_tensor(src_expr[start:end], dtype=torch.float32, device=self.device)
            dt = 1.0 / max(int(n_steps), 1)
            for step in range(int(n_steps)):
                t_flow = torch.full((x.shape[0], 1), step * dt, dtype=torch.float32, device=self.device)
                _, vel = self._predict_target_and_velocity(x, t_flow=t_flow, t_src=t_src, t_tgt=t_tgt)
                x = torch.clamp(x + vel * dt, min=0.0)
            outputs.append(x.detach().cpu().numpy().astype(np.float32))
        if not outputs:
            return np.zeros((0, src_expr.shape[1]), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _train_step(self):
        src_time, tgt_time = self._sample_training_pair()
        src_numeric = float(parse_time_from_bundle(self.bundle, src_time))
        tgt_numeric = float(parse_time_from_bundle(self.bundle, tgt_time))

        src_batch, _ = self._sample_cells(src_time, self.batch_size)
        tgt_batch, _ = self._sample_cells(tgt_time, self.batch_size)

        with torch.no_grad():
            if self.transport_pairing == "ot":
                cost = pairwise_mean_sq_dist(src_batch, tgt_batch)
                plan = sinkhorn_uniform_transport(cost, epsilon=self.ot_epsilon, max_iter=self.ot_max_iter)
                src_idx, tgt_idx = sample_transport_pairs(plan, n_samples=self.batch_size)
            else:
                src_idx = torch.randint(src_batch.shape[0], (self.batch_size,), device=self.device)
                tgt_idx = torch.randint(tgt_batch.shape[0], (self.batch_size,), device=self.device)
            paired_src = src_batch[src_idx]
            paired_tgt = tgt_batch[tgt_idx]

        t_flow = sample_t_stratified(self.batch_size, self.device)
        x_tau = (1.0 - t_flow) * paired_src + t_flow * paired_tgt
        true_vel = paired_tgt - paired_src
        pred_tgt, pred_vel = self._predict_target_and_velocity(x_tau, t_flow=t_flow, t_src=src_numeric, t_tgt=tgt_numeric)

        l_cfm = torch.mean((pred_vel - true_vel) ** 2)
        l_wass_hvg = sinkhorn_distance(pred_tgt, tgt_batch, epsilon=self.ot_epsilon, max_iter=self.ot_max_iter)
        if self.use_wass_pca_loss:
            pred_pca = self._pca_transform_torch(pred_tgt)
            tgt_pca = self._pca_transform_torch(tgt_batch)
            l_wass_pca = sinkhorn_distance(pred_pca, tgt_pca, epsilon=self.ot_epsilon, max_iter=self.ot_max_iter)
            l_wass_pca_value = float(l_wass_pca.item())
        else:
            l_wass_pca = pred_tgt.new_tensor(0.0)
            l_wass_pca_value = None
        l_mean = torch.mean((pred_tgt.mean(dim=0) - tgt_batch.mean(dim=0)) ** 2)
        l_var = self._variance_regularization_loss(pred_tgt, tgt_batch)

        total = (
            self.loss_cfm * l_cfm
            + self.loss_wass_hvg * l_wass_hvg
            + self.loss_wass_pca * l_wass_pca
            + self.loss_mean * l_mean
            + self.loss_variance * l_var
        )
        metrics = {
            "loss": float(total.item()),
            "cfm": float(l_cfm.item()),
            "wasserstein_hvg": float(l_wass_hvg.item()),
            "wasserstein_pca50": l_wass_pca_value,
            "wasserstein_pca50_enabled": bool(self.use_wass_pca_loss),
            "wasserstein_pca50_weight": float(self.loss_wass_pca),
            "mean_loss": float(l_mean.item()),
            "variance_loss": float(l_var.item()),
            "src_time": src_time,
            "tgt_time": tgt_time,
            "transport_pairing": self.transport_pairing,
            "prediction_parameterization": self.prediction_parameterization,
        }
        return total, metrics

    @torch.no_grad()
    def build_prediction_artifact(self, method_name: str, output_dir: str, max_cells_per_time: int = 0):
        self.model.eval()
        print(f"[route_a_predict] method={method_name} output_dir={output_dir}")
        pred_by_time = {}
        real_by_time = {}
        obs_names_by_time = {}
        metadata_df = self.bundle["metadata_df"]

        for time_bin in tqdm(self.bundle["eval_time_bins"], desc="predict-heldout", leave=False):
            real_idx = self.bundle["time_to_indices"][time_bin]
            if int(max_cells_per_time) > 0 and real_idx.shape[0] > int(max_cells_per_time):
                real_idx = np.random.choice(real_idx, size=int(max_cells_per_time), replace=False)
            real_expr = np.asarray(self.bundle["expression"][real_idx], dtype=np.float32)
            obs_names = metadata_df.iloc[real_idx]["obs_name"].astype(str).to_numpy()
            sources = resolve_generation_sources(self.bundle, time_bin)
            n_total = real_expr.shape[0]
            pred_parts = []
            allocated = 0
            for src_idx, source in enumerate(tqdm(sources, desc=f"sources@{time_bin}", leave=False)):
                if src_idx == len(sources) - 1:
                    n_source = n_total - allocated
                else:
                    n_source = int(round(n_total * float(source["weight"])))
                    allocated += n_source
                if n_source <= 0:
                    continue
                src_time = source["time_bin"]
                src_cells, _ = self._sample_cells(src_time, n_source)
                pred = self._rollout(
                    src_cells.detach().cpu().numpy(),
                    t_src=float(parse_time_from_bundle(self.bundle, src_time)),
                    t_tgt=float(parse_time_from_bundle(self.bundle, time_bin)),
                    n_steps=self.infer_n_steps,
                )
                pred_parts.append(pred)
            if not pred_parts:
                raise ValueError(f"failed to generate predictions for heldout time {time_bin}")
            pred_expr = np.concatenate(pred_parts, axis=0)[:n_total]
            pred_by_time[time_bin] = pred_expr.astype(np.float32, copy=False)
            real_by_time[time_bin] = real_expr.astype(np.float32, copy=False)
            obs_names_by_time[time_bin] = obs_names

        artifact = {
            "protocol_name": self.bundle["protocol_name"],
            "method_name": method_name,
            "ordered_time_bins": self.bundle["ordered_time_bins"],
            "heldout_time_bins": self.bundle["eval_time_bins"],
            "time_display_map": self.bundle.get("time_display_map", {}),
            "pred_by_time": pred_by_time,
            "real_by_time": real_by_time,
            "obs_names_by_time": obs_names_by_time,
        }
        save_prediction_artifact(output_dir, artifact)
        return output_dir

    def _validate(self, global_step: int):
        eval_dir = os.path.join(self.exp_dir, "eval_artifacts", f"step_{global_step}")
        t0 = time.time()
        print(f"[route_a_val] start validation at step={global_step}")
        artifact_dir = self.build_prediction_artifact(
            method_name="scFiT",
            output_dir=eval_dir,
            max_cells_per_time=self.validation_max_cells_per_time,
        )
        metrics = evaluate_artifact_dir(
            artifact_dir=artifact_dir,
            assets_dir=self.assets_dir,
            metadata_df=self.bundle["metadata_df"],
            hvg_gene_names=self.bundle["bundle_meta"]["hvg_gene_names"],
            epsilon=self.ot_epsilon,
            max_iter=self.ot_max_iter,
            metric_names=self.validation_metric_names,
        )
        metrics["eval_s"] = float(time.time() - t0)
        print(f"[route_a_val] done step={global_step} eval_s={metrics['eval_s']:.2f}")
        return metrics

    def train(self, epochs: int, lr: float, weight_decay: float, save_every_epochs: int, resume: str = None):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        global_step = 0
        start_epoch = 0
        if resume:
            ckpt = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            print(f"[resume] {resume} epoch={start_epoch} step={global_step}")

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        total_steps = int(epochs) * int(self.steps_per_epoch)
        overall = tqdm(total=total_steps, initial=global_step, desc="train")

        for epoch in range(start_epoch, int(epochs)):
            epoch_bar = tqdm(range(self.steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", leave=False)
            for epoch_step in epoch_bar:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                    loss, train_metrics = self._train_step()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                overall.update(1)
                wpca_metric = train_metrics.get("wasserstein_pca50")
                epoch_bar.set_postfix(
                    loss=f"{train_metrics['loss']:.4f}",
                    whvg=f"{train_metrics['wasserstein_hvg']:.4f}",
                    wpca=("off" if wpca_metric is None else f"{wpca_metric:.4f}"),
                    cfm=f"{train_metrics['cfm']:.4f}",
                    var=f"{train_metrics['variance_loss']:.4f}",
                )

                if global_step % self.log_every_steps == 0:
                    self._log(
                        {
                            "epoch": epoch + 1,
                            "epoch_step": epoch_step + 1,
                            "global_step": global_step,
                            **train_metrics,
                        }
                    )

                if global_step % self.val_every_steps == 0:
                    val_metrics = self._validate(global_step)
                    print(
                        f"\n[val step={global_step}] "
                        f"wasserstein_pca50={self._fmt(val_metrics.get('wasserstein_pca50_mean'))} "
                        f"eval_s={self._fmt(val_metrics.get('eval_s'))}"
                    )
                    self._log(
                        {
                            "epoch": epoch + 1,
                            "epoch_step": epoch_step + 1,
                            "global_step": global_step,
                            "val_wasserstein_hvg_mean": val_metrics.get("wasserstein_hvg_mean"),
                            "val_wasserstein_pca50_mean": val_metrics.get("wasserstein_pca50_mean"),
                            "val_distribution_mean_r2_mean": val_metrics.get("distribution_mean_r2_mean"),
                            "val_nn_matched_per_cell_r2_mean": val_metrics.get("nn_matched_per_cell_r2_mean"),
                            "val_cell_type_accuracy_mean": val_metrics.get("cell_type_accuracy_mean"),
                            "val_marker_mmd_mean": val_metrics.get("marker_mmd_mean"),
                            "val_eval_seconds": val_metrics.get("eval_s"),
                        }
                    )

            if (epoch + 1) % max(int(save_every_epochs), 1) == 0:
                self._save_checkpoint(epoch + 1, global_step, optimizer)

        overall.close()
        self._save_checkpoint(int(epochs), global_step, optimizer, tag="final")
        final_artifact_dir = os.path.join(self.exp_dir, "eval_artifacts", "final")
        print("[route_a_train] running final full evaluation")
        self.build_prediction_artifact(
            method_name="scFiT",
            output_dir=final_artifact_dir,
            max_cells_per_time=self.final_eval_max_cells_per_time,
        )
        final_metrics = evaluate_artifact_dir(
            artifact_dir=final_artifact_dir,
            assets_dir=self.assets_dir,
            metadata_df=self.bundle["metadata_df"],
            hvg_gene_names=self.bundle["bundle_meta"]["hvg_gene_names"],
            epsilon=self.ot_epsilon,
            max_iter=self.ot_max_iter,
        )
        self._log(
            {
                "epoch": int(epochs),
                "global_step": global_step,
                "final_wasserstein_hvg_mean": final_metrics.get("wasserstein_hvg_mean"),
                "final_wasserstein_pca50_mean": final_metrics.get("wasserstein_pca50_mean"),
                "final_distribution_mean_r2_mean": final_metrics.get("distribution_mean_r2_mean"),
                "final_nn_matched_per_cell_r2_mean": final_metrics.get("nn_matched_per_cell_r2_mean"),
                "final_cell_type_accuracy_mean": final_metrics.get("cell_type_accuracy_mean"),
                "final_marker_mmd_mean": final_metrics.get("marker_mmd_mean"),
            }
        )
        print(
            "[route_a_train] final metrics "
            f"wasserstein_hvg={self._fmt(final_metrics.get('wasserstein_hvg_mean'))} "
            f"wasserstein_pca50={self._fmt(final_metrics.get('wasserstein_pca50_mean'))} "
            f"distribution_mean_r2={self._fmt(final_metrics.get('distribution_mean_r2_mean'))}"
        )

    @staticmethod
    def _fmt(value):
        if value is None:
            return "NA"
        return f"{float(value):.4f}"


def parse_time_from_bundle(bundle: Dict, time_bin: str) -> float:
    idx = bundle["time_to_indices"][time_bin][0]
    return float(bundle["metadata_df"].iloc[idx]["author_day_numeric"])
