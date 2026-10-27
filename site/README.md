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
