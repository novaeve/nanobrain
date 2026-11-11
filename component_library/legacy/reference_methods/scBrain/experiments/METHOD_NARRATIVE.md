# scBrain Method Narrative

## Overview

**scBrain** is a compact hierarchical model for spatial transcriptomic cell annotation in diseased brain tissue. The public method name throughout the paper and release is `scBrain`; legacy implementation aliases are retained only where needed for checkpoint and log compatibility.

The main SEA-AD experiment uses the Middle Temporal Gyrus MERFISH dataset with 1,887,729 cells, a 140-gene panel, tiled spatial coordinates, and donor-stratified train/validation/test splits. The annotation task follows the Allen Brain Cell Atlas hierarchy: Class (3 labels), Subclass (24 labels), and Supertype (137 labels).

## Biological Motivation

Brain cell identity is intrinsically hierarchical. Broad neuronal and non-neuronal classes constrain possible subclasses, and subclasses constrain fine-grained supertypes. A flat classifier ignores this biological dependency and must learn coarse and fine labels from a single undifferentiated representation. scBrain instead aligns model depth with taxonomy depth: shallow features support easy coarse decisions, while deeper features are reserved for fine-grained distinctions.

Spatial context is also biologically meaningful in cortex. Laminar position, local tissue geometry, and disease progression can shift the interpretation of a transcriptional profile. For this reason, the release includes contextual scBrain variants that inject depth, morphology/spatial coordinates, or disease-conditioned FiLM modulation as controlled ablations.

## Input and Targets

The default input is the log1p-normalized `cell x gene` matrix. Spatial variants concatenate or condition on normalized x-y coordinates and optional morphology or disease covariates. The model predicts all hierarchy levels jointly and also reconstructs the input gene expression vector.

We report Macro F1, Micro F1, and one-vs-rest AUC-ROC at each level. Macro F1 is the primary metric for Supertype because rare cell types should contribute equally to the evaluation.

## Core Architecture

scBrain uses stacked feed-forward blocks rather than a token Transformer because MERFISH features are fixed gene measurements, not variable-length gene sequences. Each feed-forward block contains RMSNorm, a SwiGLU expansion, dropout, a linear projection, and a residual connection:

```text
FFN(x) = x + Dropout(W_out(SiLU(W_gate Norm(x)) * W_up Norm(x)))
```

The main model has a three-stage cascaded encoder:

```text
h1 = E1(x)                    # coarse Class representation
h2 = E2(h1)                   # Subclass representation
H  = E3(h2) -> z              # compact Supertype/reconstruction representation
```

The decoder is deliberately hierarchical rather than three independent heads:

```text
C1 = D1(h1)
C2 = D2(h2) + C1
C3 = D3(H)  + C2
```

The residual links carry coarse-level decoder features into finer levels. This lets the Supertype classifier inherit uncertainty and structure from Class and Subclass predictions without forcing premature hard decisions.

## Loss Function

The training objective combines reconstruction and hierarchy-level classification:

```text
L = L_recon + L_class + L_subclass + L_supertype
```

The reconstruction term forces the latent representation to preserve biological expression structure. This is analogous to the regularizing role of reconstruction in integration models: it discourages the classifier from exploiting label shortcuts that discard transcriptional signal.

## Model Family

| Public name | Purpose |
|-------------|---------|
| `scBrainFlat` | Single latent hierarchical-decoder baseline |
| `scBrain` | Main residual hierarchical encoder-decoder |
| `scBrainDepth` | Laminar depth injection ablation |
| `scBrainMorphoSpatial` | Cell-volume and x-y early-fusion ablation |
| `scBrainFiLM` | CPS and x-y conditioned FiLM ablation |
| `scBrainNLevel` | General N-level model for Kukanja MS/EAE |

## Main Results Summary

| Model | Supertype Macro F1 | Supertype Micro F1 | Supertype AUC |
|-------|-------------------:|-------------------:|--------------:|
| scBrainFlat | 0.6734 | 0.8579 | 0.9984 |
| scBrain, independent decoders | 0.6887 | 0.8661 | 0.9978 |
| scBrain, logit residual | 0.6997 | 0.8681 | 0.9986 |
| scBrain, feature residual, dec128 | 0.7525 | 0.8843 | 0.9992 |
| scBrain, feature residual, dec256 | 0.7611 | 0.8851 | 0.9993 |
| scBrainDepth | 0.7568 | 0.8857 | 0.9992 |
| scBrainMorphoSpatial | 0.7537 | 0.8840 | 0.9993 |
| scBrainFiLM | 0.7517 | 0.8844 | 0.9993 |
| Geneformer, full finetune | 0.7489 | 0.8786 | 0.9992 |
| scGPT (Brain), full finetune | 0.6819 | 0.8295 | 0.9963 |
| UCE, full finetune | 0.5118 | 0.7153 | 0.9948 |

The strongest scBrain configuration matches or exceeds fully fine-tuned foundation models while using a much smaller architecture and no external pretraining. The key gain comes from feature-space residual hierarchy: it improves Supertype Macro F1 substantially over independent or logit-residual decoders.

## Reproducibility Note

Use `experiments/scbrain/main.py` for SEA-AD and `experiments/kukanja/main_scbrain.py` for Kukanja. Compatibility paths are kept so that historical run identifiers, checkpoint names, and paper audit files remain traceable.
