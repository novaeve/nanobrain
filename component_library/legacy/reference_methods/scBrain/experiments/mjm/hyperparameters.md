# scBrain Hyperparameter Guide

This file documents the public scBrain configuration used in the submission. The directory name is retained for legacy compatibility; new runs should use `experiments/scbrain/main.py`.

## Checkpoint Naming

Public scBrain runs write artifacts under:

```text
experiments/scBrain/{dataset}/checkpoints/
  e{e_layers}_d{d_layers}_enc{enc_hidden_dim}_dec{dec_hidden_dim}_z{latent_dim}_s{seed}_res{residual_mode}/
```

Example:

```text
e3_d1_enc256_dec256_z20_s3028_resfeature/best_model.pth
```

Historical logs may contain legacy aliases in file names. Treat these as compatibility labels for `scBrainFlat` and `scBrain`.

## Training Configuration

| Argument | Default | Guidance |
|----------|--------:|----------|
| `--bs` | 512 | Use 256 when GPU memory is limited; 512 is the reported setting. |
| `--lr` | 1e-4 | Adam with cosine annealing; tune in `[1e-5, 5e-4]` if needed. |
| `--weight_decay` | 1e-4 | Keep fixed for reproduction. |
| `--max_epochs` | 500 | Early stopping usually triggers earlier. |
| `--patience` | 30 | Validation total loss is the stopping criterion. |
| `--dropout` | 0.3 | Used inside all SwiGLU FFN blocks. |
| `--seed` | 3028 | Submission tables report this seed unless stated otherwise. |

## Architecture Configuration

| Argument | Default | Guidance |
|----------|--------:|----------|
| `--input_dim` | 140 | SEA-AD MERFISH gene panel size. |
| `--latent_dim` | 20 | Compact biological fingerprint for reconstruction and fine labels. |
| `--enc_hidden_dim` | 256 | Encoder width; keep at 256 for reported runs. |
| `--dec_hidden_dim` | 256 | Decoder width for the main SEA-AD result; 128 is used in some ablations. |
| `--e_layers` | 3 | Number of FFN blocks in each encoder stage. |
| `--d_layers` | 1 | Shallow decoders keep classification heads lightweight. |
| `--expansion_factor` | 2.67 | SwiGLU expansion, approximately `8/3`. |
| `--output_num` | `[3,24,137]` | Class, Subclass, and Supertype counts. |

## Residual Modes

`feature` is the main model:

```text
C2 = D2(h2) + C1
C3 = D3(H)  + C2
```

This keeps coarse-level evidence in continuous feature space and gives the best Supertype Macro F1.

`logit` injects projected upstream logits:

```text
C2 = D2(h2) + proj(logits1)
C3 = D3(H)  + proj(logits2)
```

Use only for ablation, because it discretizes upstream uncertainty too early.

`none` removes cross-level residuals and evaluates independent decoders:

```text
C2 = D2(h2)
C3 = D3(H)
```

## Reproduction Commands

Main SEA-AD result:

```bash
python experiments/scbrain/main.py \
  --model_name scBrain \
  --npz_path data/global_data.npz \
  --enc_hidden_dim 256 --dec_hidden_dim 256 \
  --e_layers 3 --d_layers 1 \
  --latent_dim 20 --residual_mode feature \
  --seed 3028 --device cuda:0
```

Residual ablations:

```bash
python experiments/scbrain/main.py --model_name scBrain --residual_mode feature --seed 3028
python experiments/scbrain/main.py --model_name scBrain --residual_mode logit --seed 3028
python experiments/scbrain/main.py --model_name scBrain --residual_mode none --seed 3028
```

Decoder-width ablations:

```bash
python experiments/scbrain/main.py --model_name scBrain --dec_hidden_dim 64 --seed 3028
python experiments/scbrain/main.py --model_name scBrain --dec_hidden_dim 128 --seed 3028
python experiments/scbrain/main.py --model_name scBrain --dec_hidden_dim 256 --seed 3028
```

## Loss Weights

The default trainer uses:

```python
total_loss = loss_recon + loss_c + loss_sc + loss_st
```

`loss_recon` is an expression reconstruction term, while `loss_c`, `loss_sc`, and `loss_st` are cross-entropy losses for Class, Subclass, and Supertype. Keep these weights fixed for reproducing the paper.
