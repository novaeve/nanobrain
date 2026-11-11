# Data Placeholder

Put packed datasets here before running experiments. Required paths:

- `data/global_data.npz`: SEA-AD packed MERFISH benchmark.
- `data/global_data_v2.npz`: SEA-AD packed benchmark with auxiliary depth/volume fields for ablations.
- `data/merfish_gene_names.json`: ordered MERFISH gene list.
- `data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad`: raw SEA-AD h5ad used by selected annotation-transfer baselines.
- `data/raw/kukanja/MS_xenium_data_v5_with_images_tmap.h5ad`: optional raw Kukanja MS h5ad for foundation-model wrappers.
- `data/raw/kukanja/EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad`: optional raw Kukanja EAE h5ad for foundation-model wrappers.
- `data/kukanja/kukanja_ms.npz`: Kukanja MS packed benchmark.
- `data/kukanja/kukanja_eae.npz`: Kukanja EAE packed benchmark.

No dataset is included in the code release.
