# nanoBrain 官网首版

这是 Canchen Tech AI Group 的 nanoBrain 静态官网首版，直接读取 `site/data/leaderboard.json`。

## H20 本机预览

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/nanobrain-github/site
python3 -m http.server 4173
```

训练或演化自动发布接入后，服务端只需要重新生成 `data/leaderboard.json`，浏览器刷新即可看到新的主指标、排行榜和斩杀图数据。

## 数据契约

每条自动提交结果应包含 `run_id`、`submitted_at`、`model_name`、`task`、`metric`、`value`、`parameter_count`、`dataset`、`split_id`、`commit_sha`、`hardware` 和 `status`。`parameter_count` 是参数量–性能斩杀图的开关字段。

## 自动提交

服务运行后接受：

```bash
curl -X POST http://127.0.0.1:4173/api/submit \
  -H "Content-Type: application/json" \
  --data @receipt.json
```

也可以直接调用：

```bash
python3 scripts/submit_leaderboard.py results/receipt.json \
  --endpoint http://127.0.0.1:4173/api/submit
```

接口会把新结果追加到 `site/data/submissions.jsonl`，并原子重建 `site/data/leaderboard.json`。服务由 `systemd` 的 `nanobrain-site.service` 持续运行。
