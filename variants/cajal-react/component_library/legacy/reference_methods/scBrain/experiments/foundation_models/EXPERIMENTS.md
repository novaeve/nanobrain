# Foundation Model Baseline Experiments

## Overview
Fine-tuning foundation models (Geneformer, scGPT, UCE, Nicheformer) on SEA-AD MTG MERFISH dataset for cell type annotation.

**Dataset**: 1,887,729 cells × 140 genes, 27 donors, 3-level hierarchy (Class/Subclass/Supertype)

---

## Test Results (Inference on Existing Checkpoints)

| Model | Mode | Epoch | Supertype Macro F1 | Supertype Micro F1 | Supertype AUC (Macro) | Log |
|-------|------|-------|-------------------|-------------------|---------------------|-----|
| **Geneformer** | freeze | 29 | 0.5420 | 0.7536 | 0.9968 | `logs/geneformer_freeze_test.out` |
| **Geneformer** | full | 10 | **0.7188** | **0.8693** | **0.9991** | `logs/geneformer_full_test.out` |
| **UCE** | freeze | 56 | 0.3370 | 0.5946 | 0.9864 | `logs/uce_freeze_test.out` |
| **UCE** | full | 23 | 0.5133 | 0.7161 | 0.9945 | `logs/uce_full_test.out` |
| **scGPT (brain)** | freeze | 105 | 0.1300 | 0.4645 | 0.9525 | `logs/scgpt_brain_freeze_s3028.out` |
| **scGPT (brain)** | full | 80 | 0.6824 | 0.8295 | 0.9964 | `logs/scgpt_brain_full_test.out` |
| **scGPT (human)** | freeze | 114 | 0.2343 | 0.5395 | 0.9723 | `logs/scgpt_human_freeze_test.out` |
| **Nicheformer** | freeze | - | OOM | OOM | OOM | `logs/nicheformer_freeze_s3028.out` |
| **Nicheformer** | full | - | OOM | OOM | OOM | `logs/nicheformer_full_s3028.out` |

**Best performer**: Geneformer full (F1=0.7188, epoch 10)

---

## Resume Training Sessions

All models are now resuming training from their checkpoints in tmux sessions:

| Model | Mode | GPU | Tmux Session | Training Log | Checkpoint |
|-------|------|-----|--------------|--------------|------------|
| Geneformer | freeze | cuda:1 | `fm_gf_freeze` | `logs/geneformer_freeze_s3028.out` | `experiments/Geneformer/SEA_AD_MTP_ST/checkpoints/freeze_s3028/` |
| Geneformer | full | cuda:2 | `fm_gf_full` | `logs/geneformer_full_s3028.out` | `experiments/Geneformer/SEA_AD_MTP_ST/checkpoints/full_s3028/` |
| UCE | freeze | cuda:3 | `fm_uce_freeze` | `logs/uce_freeze_s3028.out` | `experiments/UCE/SEA_AD_MTP_ST/checkpoints/freeze_s3028/` |
| UCE | full | cuda:4 | `fm_uce_full` | `logs/uce_full_s3028.out` | `experiments/UCE/SEA_AD_MTP_ST/checkpoints/full_s3028/` |
| scGPT (brain) | full | cuda:5 | `fm_scgpt_brain_full` | `logs/scgpt_brain_full_s3028.out` | `experiments/scGPT_brain/SEA_AD_MTP_ST/checkpoints/full_s3028/` |
| scGPT (human) | freeze | cuda:6 | `fm_scgpt_human_freeze` | `logs/scgpt_human_freeze_s3028.out` | `experiments/scGPT_human/SEA_AD_MTP_ST/checkpoints/freeze_s3028/` |
| scGPT (brain) | freeze | cuda:7 | `fm_scgpt_brain_freeze` | `logs/scgpt_brain_freeze_s3028.out` | `experiments/scGPT_brain/SEA_AD_MTP_ST/checkpoints/freeze_s3028/` |

### Commands to Monitor

```bash
# List all sessions
tmux ls

# Attach to a session
tmux attach -t fm_gf_freeze

# Check training progress
tail -f logs/geneformer_freeze_s3028.out

# Check GPU usage
nvidia-smi
```

---

## Training Configuration

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=500, eta_min=1e-6)
- **Early stopping**: patience=30
- **Max epochs**: 500
- **Batch size**: 256 (Geneformer/UCE), 512 (scGPT)
- **Seed**: 3028

---

## Notes

1. **Resume logic**: Modified `src/engines/scgpt_trainer.py` to automatically load `best_model.pth` if it exists when training starts
2. **Nicheformer OOM**: Both freeze and full modes ran out of memory (75GB+ usage on 80GB GPUs). Requires batch size reduction or model optimization
3. **Test features**: All models save test set embeddings to `test_features.npz` for downstream analysis
4. **Training continues**: All sessions chain test → train, so training resumes automatically after test completes

---

## Next Steps

- Monitor training progress and collect final results when early stopping triggers
- Compare with the best scBrain checkpoint (Supertype F1=0.7611) and contextual scBrain variants
- Investigate Nicheformer memory issue (consider bs=128 or gradient checkpointing)
