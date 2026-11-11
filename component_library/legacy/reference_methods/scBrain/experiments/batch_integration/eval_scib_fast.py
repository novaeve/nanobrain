#!/usr/bin/env python
"""
Batch integration evaluation using scIB metrics (fast version).

Computes per-model:
  Batch correction: ASW_batch, Graph connectivity, PCR_batch
  Bio conservation:  ASW_label, NMI, ARI, Isolated label F1, Isolated label ASW
  Overall = 0.4 * batch + 0.6 * bio

Skips LISI (iLISI/cLISI) which is extremely slow on large datasets.
Results saved incrementally per model.

Usage:
    python experiments/batch_integration/eval_scib_fast.py [--subsample 0.1]
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

EXPERIMENT_ROOT = './experiments'

MODELS = [
    # (display_name, relative_path, embed_key)
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
    parser.add_argument('--subsample', type=float, default=0.1,
                        help='Fraction of cells to subsample (default 0.1 = ~40K cells)')
    return parser.parse_args()


def load_test_labels(npz_path='./data/global_data.npz', seed=3028):
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
    n = len(batch)
    X = raw_X if raw_X is not None else np.zeros((n, 1), dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.obs['batch'] = pd.Categorical(batch.astype(str))
    adata.obs['label'] = pd.Categorical(supertype.astype(str))
    adata.obsm['X_emb'] = embedding.astype(np.float32)
    return adata


def eval_one_model(adata_raw, adata_int, logger, model_name):
    """Compute scIB metrics individually (faster than scib.metrics.metrics)."""
    logger.info(f"  Embedding: {adata_int.obsm['X_emb'].shape}, "
                f"Batches: {adata_int.obs['batch'].nunique()}, "
                f"Labels: {adata_int.obs['label'].nunique()}")

    # Precompute neighbors + Leiden clustering
    sc.pp.neighbors(adata_int, use_rep='X_emb')
    scib.cl.cluster_optimal_resolution(adata_int, label_key='label', cluster_key='cluster',
                                       cluster_function=sc.tl.leiden)

    results = {}

    # --- Batch correction metrics ---
    logger.info("  Computing ASW_batch...")
    results['ASW_batch'] = scib.me.silhouette_batch(
        adata_int, batch_key='batch', label_key='label', embed='X_emb', verbose=False)

    logger.info("  Computing graph_conn...")
    results['graph_conn'] = scib.me.graph_connectivity(adata_int, label_key='label')

    logger.info("  Computing PCR_batch...")
    results['PCR_batch'] = scib.me.pcr_comparison(
        adata_raw, adata_int, embed='X_emb', covariate='batch', verbose=False)

    # --- Bio conservation metrics ---
    logger.info("  Computing ASW_label...")
    results['ASW_label'] = scib.me.silhouette(
        adata_int, label_key='label', embed='X_emb')

    logger.info("  Computing NMI...")
    results['NMI'] = scib.me.nmi(adata_int, cluster_key='cluster', label_key='label')

    logger.info("  Computing ARI...")
    results['ARI'] = scib.me.ari(adata_int, cluster_key='cluster', label_key='label')

    # --- Aggregate ---
    batch_score = np.nanmean([results['ASW_batch'], results['graph_conn'], results['PCR_batch']])
    bio_score = np.nanmean([results['ASW_label'], results['NMI'], results['ARI']])
    results['batch_score'] = batch_score
    results['bio_score'] = bio_score
    results['overall'] = 0.4 * batch_score + 0.6 * bio_score

    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    return results


if __name__ == '__main__':
    args = parse_args()
    log_dir = './experiments/batch_integration/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, 'scib_fast.log')

    logger.info("Loading test-set labels...")
    test_labels = load_test_labels()
    n_test = len(test_labels['y_supertype'])
    logger.info(f"Test set: {n_test} cells, "
                f"{len(np.unique(test_labels['batch_donor']))} donors")

    # Subsample
    rng = np.random.RandomState(42)
    idx = rng.choice(n_test, size=int(n_test * args.subsample), replace=False)
    idx.sort()
    test_labels = {k: v[idx] for k, v in test_labels.items()}
    n_sub = len(idx)
    logger.info(f"Subsampled to {n_sub} cells ({args.subsample*100:.0f}%)")

    # Build raw reference
    adata_raw = build_adata(
        embedding=test_labels['X_raw'],
        batch=test_labels['batch_donor'],
        supertype=test_labels['y_supertype'],
        raw_X=test_labels['X_raw'],
    )
    sc.pp.neighbors(adata_raw, use_rep='X_emb')

    # Load existing results
    csv_path = os.path.join(log_dir, 'scib_fast_results.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, index_col=0)
        done_models = set(existing_df.index)
        logger.info(f"Found {len(done_models)} previously evaluated models")
    else:
        existing_df = pd.DataFrame()
        done_models = set()

    # Evaluate each model
    for model_name, rel_path, embed_key in MODELS:
        if model_name in done_models:
            logger.info(f"SKIP (done): {model_name}")
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
            embedding = data[embed_key][idx]

            if 'supertype' in data.files and 'batch' in data.files:
                batch = data['batch'][idx]
                supertype = data['supertype'][idx]
            else:
                batch = test_labels['batch_donor']
                supertype = test_labels['y_supertype']

            adata_int = build_adata(embedding, batch, supertype, test_labels['X_raw'])
            results = eval_one_model(adata_raw, adata_int, logger, model_name)

            # Incremental save
            row = pd.DataFrame(results, index=[model_name])
            existing_df = pd.concat([existing_df, row])
            existing_df.to_csv(csv_path)
            done_models.add(model_name)

            elapsed = time.time() - t0
            logger.info(f"  Completed in {elapsed:.0f}s")

        except Exception as e:
            logger.info(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Raw baseline
    if 'Raw (log1p)' not in done_models:
        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: Raw (log1p) baseline")
        t0 = time.time()
        adata_raw_eval = adata_raw.copy()
        results = eval_one_model(adata_raw, adata_raw_eval, logger, 'Raw (log1p)')
        row = pd.DataFrame(results, index=['Raw (log1p)'])
        existing_df = pd.concat([existing_df, row])
        existing_df.to_csv(csv_path)
        logger.info(f"  Completed in {time.time()-t0:.0f}s")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH INTEGRATION SUMMARY")
    logger.info("=" * 80)
    summary = existing_df[['batch_score', 'bio_score', 'overall']].sort_values(
        'overall', ascending=False)
    logger.info(f"\n{summary.to_string()}")
    summary.to_csv(os.path.join(log_dir, 'scib_fast_overall.csv'))
    logger.info("\nDone.")
