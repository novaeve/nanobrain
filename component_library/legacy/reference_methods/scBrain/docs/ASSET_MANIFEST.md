# External Asset Manifest

This release intentionally excludes datasets and model weights. Use the following layout when preparing a runnable environment.

## Required for scBrain

| Path | Purpose |
| --- | --- |
| `data/global_data.npz` | SEA-AD packed cell-by-gene matrix, labels, donor split metadata, confidence scores. |
| `data/kukanja/kukanja_ms.npz` | Kukanja MS packed benchmark. |
| `data/kukanja/kukanja_eae.npz` | Kukanja EAE packed benchmark. |
| `data/global_data_v2.npz` | Optional SEA-AD packed benchmark with depth/volume fields for auxiliary ablations. |
| `data/merfish_gene_names.json` | MERFISH gene order for foundation-model wrappers. |

## Required for Selected Baselines

| Path | Purpose |
| --- | --- |
| `data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad` | Raw SEA-AD h5ad for CellTypist, TACCO, Geneformer, UCE, and Novae gene-name mapping. |
| `data/raw/kukanja/MS_xenium_data_v5_with_images_tmap.h5ad` | Optional raw Kukanja MS h5ad for foundation-model wrappers. |
| `data/raw/kukanja/EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad` | Optional raw Kukanja EAE h5ad for foundation-model wrappers. |
| `weights/geneformer/model.safetensors` | Geneformer pretrained weights. |
| `external/Geneformer_code_only/` | Geneformer code/config directory. |
| `weights/scgpt/brain/` and `weights/scgpt/human/` | scGPT pretrained checkpoints. |
| `weights/uce/` and `external/UCE/` | UCE model assets and code. |
| `weights/scfoundation/model.pt`, `weights/scfoundation/config.json` | scFoundation checkpoint and config. |
| `weights/stack/bc_large.ckpt`, `weights/stack/basecount_1000per_15000max.pkl` | Stack assets. |
| `weights/nicheformer/`, `external/nicheformer/src/` | Nicheformer assets and code. |
| `weights/scimilarity/model_v1.1.tar.gz` | scSimilarity pretrained asset. |

## Generated Assets

Experiment scripts may write checkpoints and extracted embeddings under their corresponding `experiments/*/.../checkpoints/` directories by default. For clean runs, redirect outputs to `outputs/` or archive them separately from this source package.
