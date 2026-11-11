# 消融实验结果

本文档记录 `mouse_neural_crest` 数据集上 Route A 消融实验的当前状态与结果。

当前消融定义：

- `Main`：`OT + direct_velocity + AdaLN`
- `Full`：`OT + target_reparam + AdaLN`
- `A1`：`random_uniform + target_reparam + AdaLN`
- `C3`：`OT + target_reparam + Concat`

主要比较指标：

- `Wasserstein-PCA50`
- `Wasserstein-HVG`
- `distribution_mean_r2`

说明：

- `Best val PCA50` 指训练日志里验证阶段的最优 `val_wasserstein_pca50_mean`
- `Best full-eval PCA50/HVG/R2` 指按最优验证步挑选 checkpoint 后，再做一次完整评估得到的结果
- `Final PCA50/HVG/R2` 指 `final checkpoint` 的完整评估结果
- `Pending` 表示尚未开始或尚未完成

## Near Extrapolation

| Variant | Pairing | Parameterization | Time Conditioning | Status | Best val PCA50 | Best full-eval PCA50 | Final PCA50 | Final HVG | Final R2 | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Main | OT | direct_velocity | AdaLN | Done | 3.6570 | Pending | 3.1383 | 0.6270 | 0.9441 | 当前主模型 |
| Full | OT | target_reparam | AdaLN | Done | 3.7958 | 2.9704 | 2.9914 | 0.6265 | 0.9769 | 当前看略优于 Main |
| A1 | random_uniform | target_reparam | AdaLN | Running / Partial | 5.3597 | Pending | Pending | Pending | Pending | 当前只跑到 `step=1080` |
| C3 | OT | target_reparam | Concat | Pending | Pending | Pending | Pending | Pending | Pending | 尚未开始 |

### 目前可读结论

- `Full` 相比 `Main`，在 `Near Extrapolation` 上目前是略有提升的
- `Final PCA50`：`2.9914 < 3.1383`
- `Final HVG`：`0.6265 < 0.6270`
- `Final R2`：`0.9769 > 0.9441`

## Interpolation

| Variant | Pairing | Parameterization | Time Conditioning | Status | Best val PCA50 | Best full-eval PCA50 | Final PCA50 | Final HVG | Final R2 | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Main | OT | direct_velocity | AdaLN | Done | 4.2662 | Pending | 3.6728 | 0.6307 | 0.9357 | 当前主模型 |
| Full | OT | target_reparam | AdaLN | Done | 4.5211 | Pending | 4.7860 | 0.6155 | 0.8483 | 当前还不优于 Main |
| A1 | random_uniform | target_reparam | AdaLN | Pending | Pending | Pending | Pending | Pending | Pending | 尚未开始 |
| C3 | OT | target_reparam | Concat | Pending | Pending | Pending | Pending | Pending | Pending | 尚未开始 |

### 目前可读结论

- `Interpolation` 上，当前 `Full` 还不能说优于 `Main`
- `Final PCA50` 明显更差：`4.7860 > 3.6728`
- `Final R2` 也更差：`0.8483 < 0.9357`
- 但 `Final HVG` 略好：`0.6155 < 0.6307`
- 这一项暂时需要等 `A1 / C3` 跑完后再整体判断

## 当前进度

已完成：

- `mouse_neural_crest_near_ablation_full_v1`
- `mouse_neural_crest_interpolation_ablation_full_v1`

进行中或部分完成：

- `mouse_neural_crest_near_ablation_a1_random_pair_v1`

尚未开始：

- `mouse_neural_crest_near_ablation_c3_concat_v1`
- `mouse_neural_crest_interpolation_ablation_a1_random_pair_v1`
- `mouse_neural_crest_interpolation_ablation_c3_concat_v1`

## 对应实验目录

- `Main / Interpolation`：`results/experiments/mouse/neural_crest_route_a_interpolation_v1`
- `Main / Near`：`results/experiments/mouse/neural_crest_route_a_near_v1`
- `Full / Interpolation`：`results/experiments/ablation/mouse_neural_crest_interpolation_ablation_full_v1`
- `Full / Near`：`results/experiments/ablation/mouse_neural_crest_near_ablation_full_v1`
- `A1 / Near`：`results/experiments/ablation/mouse_neural_crest_near_ablation_a1_random_pair_v1`
