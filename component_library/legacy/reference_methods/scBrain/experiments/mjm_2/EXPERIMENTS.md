# scBrain Contextual Ablations

This directory contains the SEA-AD contextual scBrain variants used as ablations. The directory name is legacy; public-facing names below should be used in the manuscript and release docs.

## Variants

| Public name | Context signal | Injection point |
|-------------|----------------|-----------------|
| scBrainDepth | Imputed laminar depth | Additive feature at the Subclass decoder |
| scBrainMorphoSpatial | Cell volume and x-y coordinates | Early input fusion before the encoder |
| scBrainFiLM | CPS and x-y coordinates | FiLM modulation after the first encoder stage |

All three variants inherit the same residual hierarchical decoder as the main scBrain model.

## Reported SEA-AD Results

The fair comparison uses `dec_hidden_dim=128` for the contextual variants and the matching scBrain residual baseline.

| Model | Decoder width | Stop epoch | Supertype Macro F1 | Supertype Micro F1 | Supertype AUC |
|-------|--------------:|-----------:|-------------------:|-------------------:|--------------:|
| scBrain, feature residual | 128 | 54 | 0.7525 | 0.8843 | 0.9992 |
| scBrainDepth | 128 | 60 | 0.7568 | 0.8857 | 0.9992 |
| scBrainMorphoSpatial | 128 | 94 | 0.7537 | 0.8840 | 0.9993 |
| scBrainFiLM | 128 | 51 | 0.7517 | 0.8844 | 0.9993 |

The depth ablation gives the strongest contextual gain, consistent with the cortical laminar organization of neuronal subtypes. The smaller gains from morphology/spatial fusion and CPS FiLM suggest that the 140-gene MERFISH panel already captures much of the relevant cell-identity signal, while contextual signals provide complementary but limited improvement.

## Commands

```bash
python experiments/mjm_2/main_a.py --npz_path data/global_data_v2.npz --device cuda:0
python experiments/mjm_2/main_b.py --npz_path data/global_data_v2.npz --device cuda:0
python experiments/mjm_2/main_c.py --npz_path data/global_data_v2.npz --device cuda:0
```

Use these scripts only for reproducing the ablations. New primary scBrain runs should go through `experiments/scbrain/main.py`.
