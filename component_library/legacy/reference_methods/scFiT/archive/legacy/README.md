# Legacy

这里存放旧版 `scFiT` 代码和实验资产，主要包括：

- 旧版 top-k reconstruction 主线
- temporal pair / fixed-HVG 过渡版本
- 旧版评估脚本
- 对应配置、缓存、实验输出和说明文档

这些文件被保留用于回溯和对照，但不再是主目录默认维护的运行路径。

当前主线请使用仓库根目录下的：

- `train.py`
- `eval.py`
- `route_a/`
- `configs/protocols/mouse/`
- `config_mouse_route_a_*.yaml`

## 迁移进来的主要内容

- `train_legacy.py`
- `eval_legacy.py`
- `eval_temporal_legacy.py`
- `data_legacy.py`
- `configs/`
- `scripts/`
- `assets/`
- `docs/`
- `experiments/`
- `eval_assets/`
- `eval_results/`
- `pair_cache/`
- `validation/`
