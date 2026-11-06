# Paper Run Map

This file maps manuscript result blocks to the source files included in this code release. Logs are lightweight audit artifacts only; datasets, checkpoints, and extracted features are intentionally excluded.

The public method name is `scBrain`. Some historical log paths and run ids still contain `mjm` because the experiments were executed before the final method name was fixed; these names are retained only to preserve exact source-file traceability.

## SEA-AD Main Results

Use `python experiments/extract_all_metrics.py` to parse the included SEA-AD logs.

| Result block | Primary files |
| --- | --- |
| scBrain main model | `experiments/mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec128_z20_s3028_resfeature.log` |
| Classic baselines | `experiments/sklearn_baselines/*/SEA_AD_MTP_ST/record_s3028.log` |
| Non-foundation baselines | `experiments/CellTypist/logs/record_s3028.log`, `experiments/TACCO/logs/record_s3028.log`, `experiments/GraphST/SEA_AD_MTP_ST/record_d64_e600_s3028.log`, `experiments/Novae/SEA_AD_MTP_ST/record_freeze_s3028.log`, `experiments/HEIST/SEA_AD_MTP_ST/record_*.log` |
| Foundation-model baselines | `experiments/Geneformer/SEA_AD_MTP_ST/record_*.log`, `experiments/scGPT_brain/SEA_AD_MTP_ST/record_*.log`, `experiments/scGPT_human/SEA_AD_MTP_ST/record_*.log`, `experiments/UCE/SEA_AD_MTP_ST/record_*.log`, `experiments/scFoundation/SEA_AD_MTP_ST/record_*.log`, `experiments/Nicheformer/SEA_AD_MTP_ST/record_*.log`, `experiments/Stack/SEA_AD_MTP_ST/record_*.log`, `experiments/scSimilarity/SEA_AD_MTP_ST/record_*.log` |

## Ablations and Diagnostics

| Result block | Primary files |
| --- | --- |
| Decoder residual ablations | `experiments/mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec128_z20_s3028_resnone.log`, `record_e3_d1_enc256_dec128_z20_s3028_reslogit.log`, `record_e3_d1_enc256_dec64_z20_s3028_resfeature.log`, `record_e3_d1_enc256_dec256_z20_s3028_resfeature.log` |
| Architecture variants | `experiments/mjm/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028.log`, `experiments/mjm_attn/SEA_AD_MTP_ST/record_*.log`, `experiments/mjm_2/SEA_AD_MTP_ST/record_*.log`, `experiments/mjm_2_lr/SEA_AD_MTP_ST/record_*.log` |
| Multi-objective variants | `experiments/mjm_moo/SEA_AD_MTP_ST/record_*.log` |
| Linear-probe diagnostics | `experiments/eval_lr/SEA_AD_MTP_ST/record_*.log` |
| Batch integration | `experiments/batch_integration/SEA_AD_MTP_ST/scib_fast_results.csv`, `scib_fast_overall.csv` |

## External Validation

Use `python experiments/kukanja/collect_results.py` to parse Kukanja logs.

| Dataset | Primary files |
| --- | --- |
| Kukanja MS | `experiments/kukanja/Kukanja_MS/record_*.log` |
| Kukanja EAE | `experiments/kukanja/Kukanja_EAE/record_*.log` |

## Paper Build Utilities

| Artifact | Command |
| --- | --- |
| Main result table | `python paper_tools/build_main_results_table.py` |
| Level-wise appendix | `python paper_tools/build_levelwise_appendix.py` |
| Reproducibility appendix | `python paper_tools/build_reproducibility_appendix.py` |
| Paper figures | `python paper_tools/figures/make_paper_figures.py` |
