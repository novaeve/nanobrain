# Top50k R Baseline Commands

Run from:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
```

Prepare aligned raw counts:

```bash
benchmark/run_benchmark_method.sh mouse_top50k prepare_raw
```

## One-Command Launcher

Run all four in background:

```bash
bash benchmark/run_top50k_r_baselines.sh all
```

Run one method:

```bash
bash benchmark/run_top50k_r_baselines.sh cistopic
bash benchmark/run_top50k_r_baselines.sh saver
bash benchmark/run_top50k_r_baselines.sh scimpute
bash benchmark/run_top50k_r_baselines.sh scbfa
```

Logs:

```bash
tail -f benchmark/outputs/mouse_top50k/logs/cistopic_full.log
tail -f benchmark/outputs/mouse_top50k/logs/saver_full.log
tail -f benchmark/outputs/mouse_top50k/logs/scimpute_full.log
tail -f benchmark/outputs/mouse_top50k/logs/scbfa_full.log
```

## Individual Commands

cisTopic:

```bash
nohup bash -c 'CISTOPIC_TOPICS=30 CISTOPIC_ITERATIONS=500 benchmark/run_benchmark_method.sh mouse_top50k cistopic && benchmark/eval_method.sh mouse_top50k benchmark/outputs/mouse_top50k/mouse_top50k_cistopic_eval.h5ad benchmark/outputs/mouse_top50k/mouse_top50k_cistopic.biological_eval.json && python benchmark/eval_masked_method.py --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_cistopic_eval.h5ad --output_json benchmark/outputs/mouse_top50k/mouse_top50k_cistopic.masked_eval.json' \
  > benchmark/outputs/mouse_top50k/logs/cistopic_full.log 2>&1 &
```

SAVER:

```bash
nohup bash -c 'ALLOW_HEAVY_R=1 NCORES=8 SAVER_NPRED=0 benchmark/run_benchmark_method.sh mouse_top50k saver && benchmark/eval_method.sh mouse_top50k benchmark/outputs/mouse_top50k/mouse_top50k_saver_eval.h5ad benchmark/outputs/mouse_top50k/mouse_top50k_saver.biological_eval.json && python benchmark/eval_masked_method.py --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_saver_eval.h5ad --output_json benchmark/outputs/mouse_top50k/mouse_top50k_saver.masked_eval.json' \
  > benchmark/outputs/mouse_top50k/logs/saver_full.log 2>&1 &
```

scImpute:

```bash
nohup bash -c 'ALLOW_HEAVY_R=1 NCORES=8 SCIMPUTE_K=10 SCIMPUTE_DROP_THRE=0.5 benchmark/run_benchmark_method.sh mouse_top50k scimpute && benchmark/eval_method.sh mouse_top50k benchmark/outputs/mouse_top50k/mouse_top50k_scimpute_eval.h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scimpute.biological_eval.json && python benchmark/eval_masked_method.py --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scimpute_eval.h5ad --output_json benchmark/outputs/mouse_top50k/mouse_top50k_scimpute.masked_eval.json' \
  > benchmark/outputs/mouse_top50k/logs/scimpute_full.log 2>&1 &
```

scBFA:

```bash
nohup bash -c 'ALLOW_HEAVY_R=1 SCBFA_FACTORS=30 SCBFA_MAXIT=100 SCBFA_CHUNK_SIZE=1000 benchmark/run_benchmark_method.sh mouse_top50k scbfa && benchmark/eval_method.sh mouse_top50k benchmark/outputs/mouse_top50k/mouse_top50k_scbfa_eval.h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scbfa.biological_eval.json && python benchmark/eval_masked_method.py --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scbfa_eval.h5ad --output_json benchmark/outputs/mouse_top50k/mouse_top50k_scbfa.masked_eval.json' \
  > benchmark/outputs/mouse_top50k/logs/scbfa_full.log 2>&1 &
```

SAVER, scImpute, and scBFA are forced with `ALLOW_HEAVY_R=1` because top50k is memory-heavy. cisTopic is the most appropriate top50k candidate among these four.
