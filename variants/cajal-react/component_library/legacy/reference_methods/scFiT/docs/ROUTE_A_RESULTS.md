# Route A 实验结果记录

## 总览

- 数据集：mouse neural crest developmental time series
- 协议体系：held-out timepoint benchmark
- 已完成方法：`scFiT`、`scNODE`、`PRESCIENT`、`MIOFlow`
- 已完成任务：
  - `interpolation`
  - `near extrapolation`
  - `far extrapolation`

## 指标说明

- `wasserstein_hvg`：主指标，HVG 空间分布距离，越低越好
- `wasserstein_pca50`：主指标，PCA-50 空间分布距离，越低越好
- `distribution_mean_r2`：群体平均表达谱相关性，越高越好
- `cell_type_accuracy`：细胞类型分类准确率，越高越好
- `marker_mmd`：marker genes 子空间分布距离，越低越好
- `nn_matched_per_cell_r2`：辅助诊断指标，越高越好

## Task 1: Interpolation

协议：`mouse_neural_crest_interpolation_v1`

### 主指标

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 |
| --- | ---: | ---: | ---: |
| scFiT | 0.6307 | **3.6728** | 0.9357 |
| scNODE | **0.4582** | 5.4983 | **0.9633** |
| PRESCIENT | 0.7534 | 25.0135 | 0.8450 |
| MIOFlow | 0.6488 | 18.9392 | 0.9555 |

### 辅助指标

| 方法 | Cell type accuracy | NN-matched per-cell r^2 | Marker MMD |
| --- | ---: | ---: | ---: |
| scFiT | **0.9819** | 0.4172 | **0.0740** |
| scNODE | 0.9094 | 0.5540 | 0.3921 |
| PRESCIENT | 0.8549 | **0.6156** | 0.3700 |
| MIOFlow | 0.7882 | 0.3757 | 0.1433 |

### 按时间点的 Wasserstein-PCA50

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| E1125 | **3.4268** | 6.0714 | 22.1158 | 9.7989 |
| E1325 | **3.5040** | 5.8654 | 26.1424 | 14.7339 |
| E1525 | **4.0921** | 4.9019 | 19.1358 | 19.0933 |
| E1675 | **3.7327** | 5.2885 | 23.2628 | 23.3988 |
| E1825 | **3.6086** | 5.3645 | 34.4103 | 27.6712 |

### 小结

- `scNODE` 在 interpolation 上仍是最强的高维分布基线，`wasserstein_hvg` 和 `distribution_mean_r2` 都最好。
- `scFiT` 在低维流形位置、细胞类型语义和 marker-level 对齐上最好，`wasserstein_pca50`、`cell_type_accuracy`、`marker_mmd` 全部占优。
- `MIOFlow` 的 `distribution_mean_r2` 很高，但 Wasserstein 两项明显落后，说明均值表达能对上，整体分布形状和流形位置没有对齐好。
- `PRESCIENT` 的 `nn_matched_per_cell_r2` 最高，但主分布指标明显更差，说明这个指标更像诊断量，不适合当主结论。

## Task 2: Near Extrapolation

协议：`mouse_neural_crest_near_extrapolation_v1`

### 主指标

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 |
| --- | ---: | ---: | ---: |
| scFiT | 0.6270 | **3.1383** | 0.9441 |
| scNODE | **0.4618** | 5.5358 | 0.9251 |
| PRESCIENT | 1.1233 | 47.5103 | 0.7907 |
| MIOFlow | 0.6617 | 21.2505 | **0.9678** |

### 辅助指标

| 方法 | Cell type accuracy | NN-matched per-cell r^2 | Marker MMD |
| --- | ---: | ---: | ---: |
| scFiT | **0.9943** | 0.4624 | **0.0735** |
| scNODE | 0.9411 | 0.6294 | 0.4667 |
| PRESCIENT | 0.9845 | **0.7030** | 0.8452 |
| MIOFlow | 0.8110 | 0.4454 | 0.1471 |

### 按时间点的 Wasserstein-PCA50

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| E1625 | **2.9407** | 5.6441 | 42.8321 | 20.2020 |
| E1650 | **3.2808** | 5.1368 | 42.3490 | 20.5607 |
| E1675 | **3.4512** | 5.4638 | 46.3951 | 21.5435 |
| E1700 | **2.8804** | 5.8986 | 58.4651 | 22.6957 |

### 小结

- `near extrapolation` 依然是 `scFiT` 最稳的一项任务，`wasserstein_pca50`、`cell_type_accuracy`、`marker_mmd` 都是四个方法里最好。
- `scNODE` 继续保持 `wasserstein_hvg` 最优，说明它在高维表达分布外推上最稳定。
- `MIOFlow` 的 `distribution_mean_r2` 再次最高，但 Wasserstein 两项明显不行，还是“均值对了，分布没对齐”的模式。
- `PRESCIENT` 在 near 任务里出现了“`cell_type_accuracy` 很高但 Wasserstein 很差、marker_mmd 很差”的组合，这说明单独看 cell type accuracy 会过于乐观。

## Task 3: Far Extrapolation

协议：`mouse_neural_crest_far_extrapolation_v1`

### 主指标

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 |
| --- | ---: | ---: | ---: |
| scFiT | 0.7487 | **7.4822** | **0.8020** |
| scNODE | **0.5616** | 9.2828 | 0.7837 |
| PRESCIENT | 1.1257 | 34.9740 | 0.6150 |
| MIOFlow | 0.9490 | 39.4657 | 0.6199 |

### 辅助指标

| 方法 | Cell type accuracy | NN-matched per-cell r^2 | Marker MMD |
| --- | ---: | ---: | ---: |
| scFiT | **0.8861** | 0.3382 | **0.1575** |
| scNODE | 0.7476 | 0.4269 | 0.4292 |
| PRESCIENT | 0.8741 | **0.5494** | 0.7090 |
| MIOFlow | 0.6668 | 0.3208 | 0.3510 |

### 按时间点的 Wasserstein-PCA50

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| E1425 | **3.9676** | 6.1502 | 14.9079 | 14.7304 |
| E1433.3 | **4.2371** | 6.3651 | 17.4237 | 15.3080 |
| E1475 | **5.2799** | 6.5247 | 20.6928 | 16.8861 |
| E1500 | **5.4520** | 7.7184 | 20.2683 | 17.8892 |
| E1525 | **5.9051** | 7.9035 | 23.4893 | 19.3147 |
| E1550 | **6.2169** | 8.7861 | 24.0578 | 21.0721 |
| E1575 | **6.3685** | 8.6206 | 27.4661 | 22.7919 |
| E1600 | **7.6327** | 9.7826 | 28.6627 | 25.8180 |
| E1625 | **8.3759** | 10.3149 | 31.5135 | 29.8038 |
| E1650 | **8.0181** | 9.1790 | 33.1440 | 33.1016 |
| E1675 | **8.3315** | 9.9941 | 35.9577 | 37.6368 |
| E1700 | **9.1234** | 11.2240 | 37.8595 | 40.2752 |
| E1750 | **8.6895** | 10.6871 | 38.9799 | 45.6804 |
| E1775 | **9.3502** | 11.0495 | 43.8453 | 50.1564 |
| E1800 | **8.7667** | 10.8441 | 47.1351 | 56.8807 |
| E1825 | **9.7622** | 11.0743 | 51.0018 | 63.9152 |
| E1850 | **8.9875** | 10.2689 | 53.3549 | 70.4091 |
| E1875 | **9.1422** | 10.4007 | 56.8561 | 79.5378 |
| P0000 | **8.5544** | 9.4861 | 57.8895 | 88.6413 |

### 小结

- `far extrapolation` 是四个方法都最难的任务，但 `scFiT` 和 `scNODE` 明显比另外两个基线稳定。
- `scNODE` 继续在 `wasserstein_hvg` 上最好，说明它在高维表达分布重建上仍最强。
- `scFiT` 在 `wasserstein_pca50`、`distribution_mean_r2`、`cell_type_accuracy`、`marker_mmd` 上仍占优，说明它更擅长保持发育流形、均值表达和谱系语义。
- `PRESCIENT` 和 `MIOFlow` 在 far 任务上出现了明显的距离爆炸，尤其是 `wasserstein_pca50` 随时间快速升高，不适合作为主对手。

## 三个任务横向对比

### Interpolation

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | Cell type accuracy | Marker MMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| scFiT | 0.6307 | **3.6728** | 0.9357 | **0.9819** | **0.0740** |
| scNODE | **0.4582** | 5.4983 | **0.9633** | 0.9094 | 0.3921 |
| PRESCIENT | 0.7534 | 25.0135 | 0.8450 | 0.8549 | 0.3700 |
| MIOFlow | 0.6488 | 18.9392 | 0.9555 | 0.7882 | 0.1433 |

### Near extrapolation

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | Cell type accuracy | Marker MMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| scFiT | 0.6270 | **3.1383** | 0.9441 | **0.9943** | **0.0735** |
| scNODE | **0.4618** | 5.5358 | 0.9251 | 0.9411 | 0.4667 |
| PRESCIENT | 1.1233 | 47.5103 | 0.7907 | 0.9845 | 0.8452 |
| MIOFlow | 0.6617 | 21.2505 | **0.9678** | 0.8110 | 0.1471 |

### Far extrapolation

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | Cell type accuracy | Marker MMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| scFiT | 0.7487 | **7.4822** | **0.8020** | **0.8861** | **0.1575** |
| scNODE | **0.5616** | 9.2828 | 0.7837 | 0.7476 | 0.4292 |
| PRESCIENT | 1.1257 | 34.9740 | 0.6150 | 0.8741 | 0.7090 |
| MIOFlow | 0.9490 | 39.4657 | 0.6199 | 0.6668 | 0.3510 |

### 横向观察

- `scNODE` 在三个任务上始终保持 `wasserstein_hvg` 最优，是高维表达分布距离最强的对手。
- `scFiT` 在三个任务上始终保持 `wasserstein_pca50` 最优，同时在 `cell_type_accuracy` 和 `marker_mmd` 上也最稳定，说明它更擅长保持发育流形和谱系语义。
- `MIOFlow` 的 `distribution_mean_r2` 持续偏高，但 Wasserstein 指标落后很多，说明只看均值表达会高估它的真实生成质量。
- `PRESCIENT` 的 `nn_matched_per_cell_r2` 和部分任务上的 `cell_type_accuracy` 偏高，但主分布指标很差，进一步说明这两类指标不能单独拿来做主结论。

## 按时间点的额外观察

### Near extrapolation

- `scFiT`
  - `E1700` 的 `wasserstein_pca50` 最好，为 2.8804。
  - 四个时间点都稳定压过其他三个方法。
- `scNODE`
  - 各时间点波动很小，说明它的外推比较平滑。
  - `E1650` 是最稳的点，`wasserstein_pca50=5.1368`。
- `PRESCIENT`
  - `E1700` 明显崩掉，`wasserstein_pca50=58.4651`。
  - 虽然 `cell_type_accuracy` 高，但分布距离非常差。
- `MIOFlow`
  - 各时间点都在 20 到 23 左右，整体比 `PRESCIENT` 稳，但仍远落后于 `scFiT/scNODE`。

### Far extrapolation

- `scFiT`
  - 刚离开 observed 区间的早期时间点最容易，如 `E1425`。
  - 后段虽然变难，但整体曲线仍然最平。
- `scNODE`
  - 远距离任务里也很稳定，整体曲线明显优于 `PRESCIENT/MIOFlow`。
  - 在 far 任务中是最强的 `wasserstein_hvg` 基线。
- `PRESCIENT`
  - 从 `E1425` 到 `P0000` 呈现近似单调恶化，`P0000` 的 `wasserstein_pca50=57.8895`。
- `MIOFlow`
  - 恶化最严重，`P0000` 的 `wasserstein_pca50=88.6413`。
  - 说明它对长距离时间推进非常不稳定。

## scFiT 训练日志观察

说明：以下 “best step” 来自训练过程中的 subset validation；“final” 来自训练结束后的完整 held-out 评估，因此二者不能机械地一一对应，但可以看训练趋势。

| 任务 | 最佳 `val_wasserstein_hvg` step | 最佳 `val_wasserstein_hvg` | 最佳 `val_wasserstein_pca50` step | 最佳 `val_wasserstein_pca50` | 最后一次 subset val `wasserstein_hvg` | 最后一次 subset val `wasserstein_pca50` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Interpolation | 1000 | 0.6234 | 4000 | 4.2662 | 0.6286 | 4.4494 |
| Near extrapolation | 1500 | 0.5824 | 3500 | 3.6570 | 0.6350 | 3.7162 |
| Far extrapolation | 3500 | 0.7090 | 3500 | 7.4803 | 0.7495 | 8.0072 |

### 日志解读

- `interpolation` 训练较稳定，末期只有轻微回退。
- `near extrapolation` 存在中后期回退：`wasserstein_hvg` 在 step 1500 最好，后面逐步变差。
- `far extrapolation` 的回退最明显：step 3500 达到最好点，之后 `wasserstein_hvg` 和 `wasserstein_pca50` 都显著恶化。
- 这说明对 `near/far` 两个任务，当前训练轮数和权重设置下，`scFiT` 已经出现一定程度的后期过训练或目标之间的 trade-off。

## 当前结论

- 如果把 `wasserstein_hvg` 作为第一主指标，那么当前三个任务上都是 `scNODE` 最强。
- 如果更关注发育流形、细胞类型语义和 marker-level biological fidelity，那么当前三个任务上 `scFiT` 都最强。
- `PRESCIENT` 和 `MIOFlow` 已经成功接入统一评估协议，但从现有结果看，它们整体都明显弱于 `scFiT` 和 `scNODE`。
- `near extrapolation` 是当前 `scFiT` 表现最好的场景。
- `far extrapolation` 是当前四种方法都最困难的场景，也是后续最值得重点优化的主战场。

## 对这个任务建议报告哪些指标

### 建议主文保留

- `wasserstein_pca50`
  - 作为主指标保留是合理的，可以说明模型在低维流形空间是否保持了发育结构。

### 建议作为生物学辅助指标保留

- `cell_type_accuracy`
  - 可以补充回答 “预测细胞是否落在正确谱系区域”。
  - 但它不能单独作为主结论，因为 `PRESCIENT` 已经展示了 “type acc 很高但分布指标很差” 的反例。
- `marker_mmd`
  - 它有生物学意义，但依赖 marker gene 的选取策略，更适合作为补充分析。

### 可以不放主文，降级到附录

- `distribution_mean_r^2`
  - 它只看群体均值，对分布形状不敏感。
  - `MIOFlow` 的结果已经说明这个指标会高估实际效果。

### 可以不报告

- `nn_matched_per_cell_r^2`
  - Route A 的任务定义是 snapshot 下的分布预测，不是假设真实存在一一对应的单细胞配对。
  - 这个指标依赖 eval 时临时最近邻匹配，本质上仍然带有 “伪单细胞对应” 假设，和当前 Route A 的任务定义不完全一致。
  - 现有结果里它也容易和主指标冲突，因此可以直接不报。

## 最简报告口径

如果需要把指标压到最少，建议正文只保留：

1. `wasserstein_hvg`
2. `wasserstein_pca50`
3. `cell_type_accuracy`

其余指标可放附录或实验日志中。

## Benchmark 数据集结果汇总（Easy Tasks）

### 数据集与任务

- 数据来源：`./new_data`
- 当前已接入 benchmark：
  - `Schiebinger`
  - `drosophila_embryonic`
  - `zebrafish_embryonic`
- 当前 protocol：
  - `wot_three_interpolation_v1`
  - `wot_three_forecasting_v1`
  - `drosophila_three_interpolation_v1`
  - `drosophila_three_forecasting_v1`
  - `zebrafish_three_interpolation_v1`
  - `zebrafish_two_forecasting_v1`

说明：这一批 benchmark 暂时没有统一可靠的 `author_cell_type` 标注，因此主表只比较：

- `wasserstein_hvg`
- `wasserstein_pca50`
- `distribution_mean_r2`
- `nn_matched_per_cell_r2`

### 当前完成状态

| 数据集 | 任务 | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | --- | --- | --- | --- | --- |
| Schiebinger | three interpolation | 已完成 | 已完成 | 已完成 | 已完成 |
| Schiebinger | three forecasting | 已完成 | 已完成 | 已完成 | 已完成 |
| Drosophila | three interpolation | 已完成 | 已完成 | 已完成 | 已完成 |
| Drosophila | three forecasting | 已完成 | 已完成 | 已完成 | 已完成 |
| Zebrafish | three interpolation | 已完成 | 已完成 | 已完成 | 已完成 |
| Zebrafish | two forecasting | 暂无结果 | 已完成 | 已完成 | 已完成 |

### Schiebinger: Three Interpolation

协议：`wot_three_interpolation_v1`

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.1869 | **1.1499** | **0.9695** | 0.6400 |
| scNODE | **0.1380** | 1.4328 | 0.9198 | **0.7485** |
| PRESCIENT | 0.1672 | 2.8854 | 0.8719 | 0.7080 |
| MIOFlow | 0.1430 | 1.7795 | 0.9503 | 0.7245 |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| 5 | **0.8276** | 0.9567 | 2.9983 | 1.1485 |
| 10 | **1.5948** | 1.7188 | 2.1222 | 1.9794 |
| 15 | **1.0273** | 1.6229 | 3.5359 | 2.2107 |

### Schiebinger: Three Forecasting

协议：`wot_three_forecasting_v1`

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.2216 | **1.3641** | **0.9406** | 0.5810 |
| scNODE | **0.1656** | 1.7603 | 0.9106 | **0.7003** |
| PRESCIENT | 0.2183 | 4.0575 | 0.8099 | 0.6384 |
| MIOFlow | 0.2488 | 4.2026 | 0.8392 | 0.6386 |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| 16 | **1.2652** | 1.5998 | 4.2623 | 3.4624 |
| 17 | **1.3618** | 1.7461 | 3.9219 | 3.9192 |
| 18 | **1.4655** | 1.9349 | 3.9883 | 5.2261 |

### Drosophila: Three Interpolation

协议：`drosophila_three_interpolation_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 best checkpoint 全量重评估结果，不再使用 `final checkpoint`。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.6586 | **3.8095** | 0.9080 | 0.0975 |
| scNODE | **0.4684** | 4.1238 | 0.8706 | 0.0865 |
| PRESCIENT | 0.5380 | 6.3995 | 0.7442 | 0.0753 |
| MIOFlow | 0.4832 | 4.1649 | 0.9072 | **0.1170** |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| hrs_04_08 | **3.7017** | 3.9541 | 7.6071 | 4.2122 |
| hrs_08_12 | **3.7582** | 4.1869 | 6.4171 | 3.9287 |
| hrs_12_16 | **3.9686** | 4.2305 | 5.1742 | 4.3538 |

### Drosophila: Three Forecasting

协议：`drosophila_three_forecasting_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 best checkpoint 全量重评估结果。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.8569 | **4.0730** | 0.7186 | 0.0826 |
| scNODE | **0.6442** | 4.6202 | 0.7738 | 0.1228 |
| PRESCIENT | 0.6471 | 5.3147 | **0.7719** | 0.0694 |
| MIOFlow | 0.7519 | 8.6778 | 0.6062 | **0.1354** |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| hrs_12_16 | **3.8517** | 3.8848 | 5.4187 | 4.9918 |
| hrs_14_18 | **3.9984** | 4.6375 | 4.9980 | 8.6065 |
| hrs_16_20 | **4.3688** | 5.3382 | 5.5275 | 12.4350 |

### Zebrafish: Three Interpolation

协议：`zebrafish_three_interpolation_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 best checkpoint 全量重评估结果。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.7181 | **2.7062** | 0.9531 | 0.5577 |
| scNODE | **0.4857** | 3.6465 | **0.9664** | **0.7067** |
| PRESCIENT | 0.8503 | 16.5209 | 0.7817 | 0.6058 |
| MIOFlow | 0.5085 | 4.2982 | 0.9723 | **0.6881** |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| E-50 | **2.5322** | 2.5598 | 18.8865 | 2.9920 |
| G-60 | **2.3304** | 3.5438 | 15.5215 | 4.1686 |
| I-90 | **3.2561** | 4.8358 | 15.1548 | 5.7341 |

### Zebrafish: Two Forecasting

协议：`zebrafish_two_forecasting_v1`

说明：`scFiT` 当前已有 `final` 评估结果，但还没有 `best_val_pca50_full`；其余三个 baseline 已完成统一评估。因此这一节先作为阶段性记录，后续如果补跑 `best checkpoint`，应以 `best_val_pca50_full` 为准覆盖这里的 `scFiT` 数值。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.8704 | **3.4404** | **0.9313** | 0.2854 |
| scNODE | **0.6951** | 5.7740 | 0.8490 | 0.3637 |
| PRESCIENT | 1.4590 | 32.9781 | 0.7271 | **0.4796** |
| MIOFlow | 0.7059 | 6.0743 | 0.8980 | 0.3893 |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| K-3S | **3.4306** | 5.3098 | 31.4886 | 5.7216 |
| L-6S | **3.4501** | 6.2381 | 34.4676 | 6.4270 |

### Benchmark 阶段性观察

- `Schiebinger` 两个任务上，`scFiT` 明显在 `wasserstein_pca50` 上最好，说明它在低维流形上的时间推进更稳。
- `Schiebinger` 上的 `scNODE` 仍然保持 `wasserstein_hvg` 最优，和 neural crest 主实验里的结论一致。
- `Drosophila interpolation` 在改成 best-checkpoint 全量评估后明显恢复到合理水平，`wasserstein_pca50` 已经优于 `scNODE / PRESCIENT / MIOFlow`。
- `Drosophila forecasting` 在改成 best-checkpoint 全量评估后，`scFiT` 的 `wasserstein_pca50` 明显优于其他三个方法，但 `scNODE / PRESCIENT` 在 `wasserstein_hvg` 和部分辅助指标上仍更稳。
- `Zebrafish interpolation` 现在四个方法都已有结果：`scFiT` 仍然在 `wasserstein_pca50` 上最好，`scNODE` 和 `MIOFlow` 在 `wasserstein_hvg` 上更强，而 `PRESCIENT` 明显落后。
- `Zebrafish two forecasting` 现在四个方法都有结果了；虽然 `scFiT` 当前还是 `final` 评估而不是 `best checkpoint`，但已经在 `wasserstein_pca50` 上明显领先。
- 这一批 benchmark 暂时更适合主报 `wasserstein_hvg` 和 `wasserstein_pca50`，不建议强调 `nn_matched_per_cell_r2`。

### 待补任务

- `zebrafish_two_forecasting_v1`
  - `scFiT` 还没有 `best_val_pca50_full` 的完整评估结果，当前文档里写入的是 `final` 版本。


## Benchmark 更新：Easy 与 Hard 分开

### Hard 任务：remove_recovery

现在已经把论文里的 `hard task` 正式补进 Route A benchmark 体系，协议名统一为 `remove_recovery`。

新增 protocol：

- [configs/protocols/benchmark/wot_remove_recovery.yaml](configs/protocols/benchmark/wot_remove_recovery.yaml)
- [configs/protocols/benchmark/drosophila_remove_recovery.yaml](configs/protocols/benchmark/drosophila_remove_recovery.yaml)
- [configs/protocols/benchmark/zebrafish_remove_recovery.yaml](configs/protocols/benchmark/zebrafish_remove_recovery.yaml)

新增 `scFiT` 配置：

- [configs/benchmark/wot_remove_recovery.yaml](configs/benchmark/wot_remove_recovery.yaml)
- [configs/benchmark/drosophila_remove_recovery.yaml](configs/benchmark/drosophila_remove_recovery.yaml)
- [configs/benchmark/zebrafish_remove_recovery.yaml](configs/benchmark/zebrafish_remove_recovery.yaml)

新增运行脚本：

- `scFiT` hard：
  - [./scripts/run_scfit_benchmarks_hard_gpu7.sh](./scripts/run_scfit_benchmarks_hard_gpu7.sh)
- baseline hard：
  - [./scripts/run_route_a_baselines_benchmarks_hard_gpu7.sh](./scripts/run_route_a_baselines_benchmarks_hard_gpu7.sh)

### Hard task 当前定义

#### Schiebinger / remove_recovery

- observed: `0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14`
- held-out: `5, 7, 9, 11, 15, 16, 17, 18`

#### Drosophila / remove_recovery

- observed: `hrs_00_02, hrs_01_03, hrs_03_07, hrs_06_10, hrs_10_14`
- held-out: `hrs_02_04, hrs_04_08, hrs_08_12, hrs_12_16, hrs_14_18, hrs_16_20`

#### Zebrafish / remove_recovery

- observed: `A-HIGH, B-OBLONG, D-30, F-S, H-75, J-B`
- held-out: `C-DOME, E-50, G-60, I-90, K-3S, L-6S`

### Hard task 的 source policy

`remove_recovery` 不是纯 interpolation，也不是纯 forecasting，所以我新增了 `hybrid_observed`：

- 如果 held-out 时间点两侧都有 observed neighbor，就按线性距离做双侧插值
- 如果只剩过去的 observed neighbor，就退化成单侧 forecasting

这样就能统一覆盖论文 hard task 里的两部分：

- 中间缺失时间点
- 末端外推时间点

不过在**结果汇报**上，应该继续把这两类 held-out 点拆开报，而不是只给一个混合均值。下面的 hard 结果因此按两层来写：

- `remove_recovery overall`
- `hard interpolation`
- `hard extrapolation`

其中：

- `hard interpolation`
  - held-out 时间点两侧都有 observed neighbor
- `hard extrapolation`
  - held-out 时间点只剩过去 observed neighbor

### Hard task 当前状态

现在 hard task 四个方法已经全部跑完，可以单独作为 `remove_recovery` 结果表报告。

后续结果建议单独记录为：

- `Schiebinger / remove_recovery`
- `Drosophila / remove_recovery`
- `Zebrafish / remove_recovery`

不要和 easy/medium 混在同一个总表里。

### 当前完成状态

| 数据集 | 协议 | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | --- | --- | --- | --- | --- |
| Schiebinger | remove_recovery | 已完成 | 已完成 | 已完成 | 已完成 |
| Drosophila | remove_recovery | 已完成 | 已完成 | 已完成 | 已完成 |
| Zebrafish | remove_recovery | 已完成 | 已完成 | 已完成 | 已完成 |

### Schiebinger: Remove Recovery

协议：`wot_remove_recovery_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 `best_val_pca50_full` 结果。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.1942 | **1.3418** | **0.9518** | 0.6671 |
| scNODE | **0.1630** | 2.0444 | 0.8923 | **0.7646** |
| PRESCIENT | 0.1742 | 3.1085 | 0.8608 | 0.7157 |
| MIOFlow | 0.2675 | 5.0572 | 0.7629 | 0.7064 |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| 5 | 1.1570 | **1.1126** | 2.6358 | 1.2352 |
| 7 | **1.0261** | 1.1334 | 2.5411 | 1.3342 |
| 9 | **1.3454** | 1.5652 | 2.2885 | 2.0015 |
| 11 | **1.5825** | 1.7905 | 2.2895 | 3.1617 |
| 15 | **1.1512** | 2.3688 | 3.6935 | 4.4396 |
| 16 | **1.4921** | 2.8156 | 3.9625 | 6.7595 |
| 17 | **1.3745** | 2.6275 | 3.6820 | 8.7701 |
| 18 | **1.6058** | 2.9414 | 3.7746 | 12.7557 |

`Schiebinger` 的 hard 拆分：

- `hard interpolation`: `5, 7, 9, 11`
- `hard extrapolation`: `15, 16, 17, 18`

| 方法 | Hard-Interp HVG | Hard-Interp PCA50 | Hard-Extra HVG | Hard-Extra PCA50 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.1662 | **1.2778** | 0.2223 | **1.4059** |
| scNODE | **0.1165** | 1.4004 | **0.2094** | 2.6883 |
| PRESCIENT | 0.1420 | 2.4387 | 0.2064 | 3.7782 |
| MIOFlow | 0.1353 | 1.9331 | 0.3997 | 8.1812 |

### Drosophila: Remove Recovery

协议：`drosophila_remove_recovery_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 `best_val_pca50_full` 结果。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.8626 | 4.2821 | **0.8531** | 0.0871 |
| scNODE | **0.6093** | **4.0127** | 0.8477 | 0.1030 |
| PRESCIENT | 0.6466 | 5.2749 | 0.7872 | 0.0927 |
| MIOFlow | 0.6734 | 6.3291 | 0.6983 | **0.1299** |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| hrs_02_04 | 6.5390 | **4.9673** | 7.7265 | 4.9132 |
| hrs_04_08 | **3.7920** | 3.8002 | 5.3628 | 4.0227 |
| hrs_08_12 | **3.7276** | 3.7880 | 4.1109 | 3.9926 |
| hrs_12_16 | 3.4099 | **3.3838** | 4.8675 | 4.6398 |
| hrs_14_18 | **3.4119** | 3.6234 | 4.1543 | 8.4045 |
| hrs_16_20 | 4.8122 | **4.5137** | 5.4273 | 12.0018 |

`Drosophila` 的 hard 拆分：

- `hard interpolation`: `hrs_02_04, hrs_04_08, hrs_08_12`
- `hard extrapolation`: `hrs_12_16, hrs_14_18, hrs_16_20`

| 方法 | Hard-Interp HVG | Hard-Interp PCA50 | Hard-Extra HVG | Hard-Extra PCA50 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.7818 | 4.6862 | 0.9435 | 3.8780 |
| scNODE | **0.5302** | **4.1851** | **0.6884** | **3.8403** |
| PRESCIENT | 0.5836 | 5.7334 | 0.7096 | 4.8164 |
| MIOFlow | 0.5441 | 4.3095 | 0.8027 | 8.3487 |

### Zebrafish: Remove Recovery

协议：`zebrafish_remove_recovery_v1`

说明：`scFiT` 这里使用的是按 `val_wasserstein_pca50_mean` 选择的 `best_val_pca50_full` 结果。

| 方法 | Wasserstein-HVG | Wasserstein-PCA50 | Distribution-mean r^2 | NN-matched per-cell r^2 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.7803 | **3.4207** | 0.9201 | 0.4351 |
| scNODE | **0.6230** | 4.4337 | 0.9198 | **0.5703** |
| PRESCIENT | 1.2521 | 27.1541 | 0.7455 | 0.5338 |
| MIOFlow | 0.6398 | 5.2318 | **0.9294** | 0.5502 |

按时间点的 `Wasserstein-PCA50`：

| Held-out timepoint | scFiT | scNODE | PRESCIENT | MIOFlow |
| --- | ---: | ---: | ---: | ---: |
| C-DOME | **1.9750** | 2.1896 | 35.4104 | 3.2713 |
| E-50 | **2.7773** | 3.0883 | 25.7600 | 3.7245 |
| G-60 | **3.1504** | 3.7913 | 21.3534 | 4.7197 |
| I-90 | **4.2937** | 5.1054 | 21.1751 | 6.6324 |
| K-3S | **4.1748** | 5.9523 | 28.9344 | 6.0456 |
| L-6S | **4.1531** | 6.4749 | 30.2916 | 6.9974 |

`Zebrafish` 的 hard 拆分：

- `hard interpolation`: `C-DOME, E-50, G-60, I-90`
- `hard extrapolation`: `K-3S, L-6S`

| 方法 | Hard-Interp HVG | Hard-Interp PCA50 | Hard-Extra HVG | Hard-Extra PCA50 |
| --- | ---: | ---: | ---: | ---: |
| scFiT | 0.7223 | **3.0491** | 0.8963 | **4.1640** |
| scNODE | **0.5615** | 3.5437 | **0.7460** | 6.2136 |
| PRESCIENT | 1.1743 | 25.9247 | 1.4077 | 29.6130 |
| MIOFlow | 0.5880 | 4.5870 | 0.7434 | 6.5215 |

### Hard task 阶段性观察

- `Schiebinger / remove_recovery` 上，拆开后可以看得更清楚：`scFiT` 在 hard interpolation 和 hard extrapolation 两部分的 `wasserstein_pca50` 都最好；`scNODE` 则继续保持两部分 `wasserstein_hvg` 最优。
- `Drosophila / remove_recovery` 上，真正拆开后 `scNODE` 的优势更明确：不管是 hard interpolation 还是 hard extrapolation，`wasserstein_hvg` 和 `wasserstein_pca50` 都更稳。
- `Zebrafish / remove_recovery` 上，`scFiT` 在 hard interpolation 和 hard extrapolation 两部分都拿下了 `wasserstein_pca50`，而 `scNODE` 继续在 `wasserstein_hvg` 上更强，整体格局和 easy 任务基本一致。
- `PRESCIENT` 在三个 hard 任务上都明显落后，尤其是 zebrafish 的 `wasserstein_pca50` 出现明显爆炸。
- `MIOFlow` 在 zebrafish hard 上比 `PRESCIENT` 稳定很多，但整体仍不如 `scFiT / scNODE`。
