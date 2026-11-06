#!/usr/bin/env python
"""
Batch integration evaluation using scIB metrics.

Evaluates all trained models' cell representations on the SEA-AD-MTP-ST test set
(6 held-out donors, 405,692 cells).

Metrics computed:
  Batch correction: ASW_batch, Graph connectivity, iLISI, PCR_batch
  Bio conservation: ASW_label, NMI, ARI, cLISI, Isolated label F1, Isolated label ASW
  Overall = 0.4 * batch + 0.6 * bio

Incremental: results are saved per-model so partial runs are not lost.

Usage:
    python experiments/batch_integration/eval_scib.py [--subsample 0.5]
"""
import sys, os, argparse, time, traceback
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import numpy as np
import anndata as ad
import scanpy as sc
import scib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger


# ── All models with pre-computed test_features.npz ──────────────────────────
EXPERIMENT_ROOT = './experiments'

MODELS = [
    # (display_name, relative_path_to_test_features_dir, embed_key)
    # --- MJM family ---
    ('mjm (origin)',           'mjm/SEA_AD_MTP_ST/checkpoints_origin',                                    'latent'),
    ('mjm (dec256)',           'mjm/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028',              'latent'),
    ('mjm_1 (feat,dec128)',    'mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resfeature', 'latent'),
    ('mjm_1 (logit,dec128)',   'mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_reslogit',   'latent'),
    ('mjm_1 (none,dec128)',    'mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resnone',    'latent'),
    ('mjm_1 (feat,dec256)',    'mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028_resfeature', 'latent'),
    ('mjm_1 (feat,dec64)',     'mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec64_z20_s3028_resfeature',  'latent'),
    ('mjm_2a (depth)',         'mjm_2/SEA_AD_MTP_ST/checkpoints/a_e3_d1_enc256_dec128_z20_s3028_resfeature_ldepth0.1', 'latent'),
    ('mjm_2b (vol)',           'mjm_2/SEA_AD_MTP_ST/checkpoints/b_e3_d1_enc256_dec128_z20_s3028_resfeature_lvol0.1',   'latent'),
    ('mjm_2c (FiLM)',          'mjm_2/SEA_AD_MTP_ST/checkpoints/c_e3_d1_enc256_dec128_z20_s3028_resfeature_film64',    'latent'),
    ('mjm_2a_lr (depth)',      'mjm_2_lr/SEA_AD_MTP_ST/checkpoints/a_e3_enc256_z20_s3028_ldepth0.1',      'latent'),
    ('mjm_2c_lr (FiLM)',       'mjm_2_lr/SEA_AD_MTP_ST/checkpoints/c_e3_enc256_z20_s3028_film64',         'latent'),
    ('mjm_lr',                 'mjm_lr/SEA_AD_MTP_ST/checkpoints/mjm_lr_e3_enc256_z20_s3028',             'latent'),
    ('mjm_1_lr',               'mjm_lr/SEA_AD_MTP_ST/checkpoints/mjm_1_lr_e3_enc256_z20_s3028',           'latent'),
    ('mjm_attn',               'mjm_attn/SEA_AD_MTP_ST/checkpoints/dec_e3_d1_enc256_dec256_z20_h8_bs4096_s3028', 'latent'),
    # --- scGPT ---
    ('scGPT (scratch)',        'scGPT/SEA_AD_MTP_ST/checkpoints',                          'cell_emb'),
    ('scGPT_brain (freeze)',   'scGPT_brain/SEA_AD_MTP_ST/checkpoints/freeze_s3028',       'cell_emb'),
    ('scGPT_brain (full)',     'scGPT_brain/SEA_AD_MTP_ST/checkpoints/full_s3028',         'cell_emb'),
    ('scGPT_human (freeze)',   'scGPT_human/SEA_AD_MTP_ST/checkpoints/freeze_s3028',       'cell_emb'),
    ('scGPT_human (full)',     'scGPT_human/SEA_AD_MTP_ST/checkpoints/full_s3028',         'cell_emb'),
    # --- Foundation models ---
    ('Geneformer (freeze)',    'Geneformer/SEA_AD_MTP_ST/checkpoints/freeze_s3028',        'cell_emb'),
    ('Geneformer (full)',      'Geneformer/SEA_AD_MTP_ST/checkpoints/full_s3028',          'cell_emb'),
    ('UCE (freeze)',           'UCE/SEA_AD_MTP_ST/checkpoints/freeze_s3028',               'cell_emb'),
    ('UCE (full)',             'UCE/SEA_AD_MTP_ST/checkpoints/full_s3028',                 'cell_emb'),
    ('Stack (freeze)',         'Stack/SEA_AD_MTP_ST/checkpoints/freeze_s3028',             'cell_emb'),
    ('Stack (full)',           'Stack/SEA_AD_MTP_ST/checkpoints/full_s3028',               'cell_emb'),
    ('scSimilarity (full)',    'scSimilarity/SEA_AD_MTP_ST/checkpoints/full_s3028',        'cell_emb'),
    ('HEIST (freeze)',         'HEIST/SEA_AD_MTP_ST/checkpoints/freeze_s3028',             'cell_emb'),
    # --- Spatial ---
    ('Spatial GNN',            'spatial_gnn/SEA_AD_MTP_ST/checkpoints/k15_gcn2_h256_z128_s3028', 'latent'),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsample', type=float, default=0.5,
                        help='Fraction of cells to subsample for scIB (0.5 = 200K cells)')
    return parser.parse_args()


def load_test_labels(npz_path='./data/global_data.npz', seed=3028):
    """Load test-set labels from global_data.npz using same donor split."""
    data = np.load(npz_path, allow_pickle=True)
    batch_all = data['batch_donor']
    unique_donors = np.unique(batch_all)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_donors)
    test_donors = unique_donors[:6]
    test_mask = np.isin(batch_all, test_donors)
    return {
        'y_supertype': data['y_supertype'][test_mask],
        'batch_donor': batch_all[test_mask],
        'X_raw': np.log1p(data['X'][test_mask].astype(np.float32)),
    }


def build_adata(embedding, batch, supertype, raw_X=None):
    """Build AnnData for scIB evaluation."""
    n = len(batch)
    X = raw_X if raw_X is not None else np.zeros((n, 1), dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.obs['batch'] = pd.Categorical(batch.astype(str))
    adata.obs['label'] = pd.Categorical(supertype.astype(str))
    adata.obsm['X_emb'] = embedding.astype(np.float32)
    return adata


def compute_scib_metrics(adata_raw, adata_int, logger, model_name):
    """Compute scIB metrics and return as dict."""
    logger.info(f"  Embedding shape: {adata_int.obsm['X_emb'].shape}")
    logger.info(f"  Batches: {adata_int.obs['batch'].nunique()}, "
                f"Labels: {adata_int.obs['label'].nunique()}")

    sc.pp.neighbors(adata_int, use_rep='X_emb')

    results = scib.metrics.metrics(
        adata_raw, adata_int,
        batch_key='batch', label_key='label',
        embed='X_emb',
        cluster_key='cluster',
        silhouette_=True,
        pcr_=True,
        graph_conn_=True,
        ilisi_=True, clisi_=True,
        nmi_=True, ari_=True,
        isolated_labels_f1_=True,
        isolated_labels_asw_=True,
        type_='embed',
        n_cores=8,
        verbose=False,
    )

    for metric_name, value in results.T.iloc[:, 0].items():
        if pd.notna(value):
            logger.info(f"  {metric_name}: {value:.4f}")

    return results


if __name__ == '__main__':
    args = parse_args()
    log_dir = './experiments/batch_integration/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, 'scib_metrics.log')

    # 1. Load ground-truth test labels
    logger.info("Loading test-set labels...")
    test_labels = load_test_labels()
    n_test = len(test_labels['y_supertype'])
    logger.info(f"Test set: {n_test} cells, "
                f"{len(np.unique(test_labels['batch_donor']))} donors")

    # 2. Subsample for speed
    if args.subsample < 1.0:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_test, size=int(n_test * args.subsample), replace=False)
        idx.sort()
        test_labels = {k: v[idx] for k, v in test_labels.items()}
        n_sub = len(idx)
        logger.info(f"Subsampled to {n_sub} cells ({args.subsample*100:.0f}%)")
    else:
        idx = None
        n_sub = n_test

    # 3. Build unintegrated reference
    adata_raw = build_adata(
        embedding=test_labels['X_raw'],
        batch=test_labels['batch_donor'],
        supertype=test_labels['y_supertype'],
        raw_X=test_labels['X_raw'],
    )
    sc.pp.neighbors(adata_raw, use_rep='X_emb')

    # 4. Load existing incremental results
    csv_path = os.path.join(log_dir, 'scib_results.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, index_col=0)
        done_models = set(existing_df.index)
        logger.info(f"Found {len(done_models)} previously evaluated models, skipping them")
    else:
        existing_df = None
        done_models = set()

    # 5. Evaluate each model
    new_results = []
    for model_name, rel_path, embed_key in MODELS:
        if model_name in done_models:
            logger.info(f"SKIP (already done): {model_name}")
            continue

        feat_path = os.path.join(EXPERIMENT_ROOT, rel_path, 'test_features.npz')
        if not os.path.exists(feat_path):
            logger.info(f"SKIP (no features): {model_name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        t0 = time.time()

        try:
            data = np.load(feat_path, allow_pickle=True)
            embedding = data[embed_key]

            if 'supertype' in data.files and 'batch' in data.files:
                batch = data['batch']
                supertype = data['supertype']
            else:
                batch = test_labels['batch_donor']
                supertype = test_labels['y_supertype']

            # Subsample embedding if needed
            if idx is not None:
                embedding = embedding[idx]
                if len(batch) > n_sub:
                    batch = batch[idx]
                    supertype = supertype[idx]

            assert len(embedding) == n_sub, \
                f"{model_name}: expected {n_sub} cells, got {len(embedding)}"

            adata_int = build_adata(embedding, batch, supertype, test_labels['X_raw'])
            results = compute_scib_metrics(adata_raw, adata_int, logger, model_name)
            results.index = [model_name]
            new_results.append(results)

            # Incremental save
            all_dfs = [existing_df] if existing_df is not None else []
            all_dfs.extend(new_results)
            combined = pd.concat(all_dfs)
            combined.to_csv(csv_path)

            elapsed = time.time() - t0
            logger.info(f"  Done in {elapsed:.0f}s")

        except Exception as e:
            logger.info(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # 6. Evaluate raw baseline
    if 'Raw (log1p)' not in done_models:
        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: Raw (log1p) baseline")
        t0 = time.time()
        adata_raw_eval = adata_raw.copy()
        results_raw = compute_scib_metrics(adata_raw, adata_raw_eval, logger, 'Raw (log1p)')
        results_raw.index = ['Raw (log1p)']
        new_results.append(results_raw)
        logger.info(f"  Done in {time.time()-t0:.0f}s")

    # 7. Final combined save
    all_dfs = [existing_df] if existing_df is not None else []
    all_dfs.extend(new_results)
    if all_dfs:
        df = pd.concat(all_dfs)
        df.to_csv(csv_path)
        logger.info(f"\nAll results saved to: {csv_path}")

        # Overall scores
        batch_metrics = ['ASW_label/batch', 'PCR_batch', 'graph_conn', 'iLISI']
        bio_metrics = ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label',
                       'cLISI', 'isolated_label_F1', 'isolated_label_silhouette']
        batch_cols = [c for c in batch_metrics if c in df.columns]
        bio_cols = [c for c in bio_metrics if c in df.columns]

        if batch_cols and bio_cols:
            df['batch_score'] = df[batch_cols].mean(axis=1)
            df['bio_score'] = df[bio_cols].mean(axis=1)
            df['overall'] = 0.4 * df['batch_score'] + 0.6 * df['bio_score']
            score_df = df[['batch_score', 'bio_score', 'overall']].sort_values(
                'overall', ascending=False)
            logger.info("\n--- Overall Scores ---")
            logger.info(f"\n{score_df.to_string()}")
            score_df.to_csv(os.path.join(log_dir, 'scib_overall_scores.csv'))

    logger.info("\nDone.")
