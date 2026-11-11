#!/usr/bin/env python
"""
GraphST experiment: spatial-aware unsupervised embeddings + sklearn LR.

GraphST learns spatial-aware cell representations via graph self-supervised
contrastive learning (per-donor, since each donor is a tissue section).
Embeddings are then used to train sklearn LogisticRegression for
Class / Subclass / Supertype annotation.

Key adaptations for SEA-AD scale (1.9M cells, up to 200k per donor):
  - Fully sparse adjacency and neighbor matrices (KNN-based)
  - Per-donor GraphST training (natural for the method)
  - Sparse GCN encoder path

Usage:
    cd .
    conda activate stack
    python experiments/GraphST/main.py --device cuda:2 --dim_output 64 --epochs 600
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from src.utils.logger import get_logger

# ── Sparse GraphST components ─────────────────────────────────────────────

def sparse_mx_to_torch(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    coo = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))


def build_sparse_adj_and_neigh(spatial_coords, n_neighbors=3):
    """
    Build sparse adjacency and neighbor matrices using KNN.
    Returns scipy sparse CSR matrices (not dense).
    """
    n = spatial_coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree',
                            n_jobs=-1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    # Build sparse interaction matrix
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices[:, 1:].flatten()  # skip self
    data = np.ones(len(rows), dtype=np.float32)
    interaction = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # graph_neigh = interaction (directed KNN)
    graph_neigh = interaction.copy()

    # adj = symmetrized interaction
    adj = interaction + interaction.T
    adj.data = np.clip(adj.data, 0, 1)

    return adj, graph_neigh


def normalize_sparse_adj(adj):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}, plus self-loop."""
    adj = sp.coo_matrix(adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    return D.dot(adj_hat).dot(D).tocsr()


# ── Sparse Encoder ─────────────────────────────────────────────────────────

class SparseAvgReadout(nn.Module):
    """AvgReadout that works with sparse neighbor matrix."""
    def forward(self, emb, mask_sparse):
        # mask_sparse: sparse [N, N], emb: dense [N, D]
        vsum = torch.spmm(mask_sparse, emb)
        row_sum = torch.sparse.sum(mask_sparse, dim=1).to_dense().unsqueeze(1)
        row_sum = row_sum.clamp(min=1e-12)
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        return torch.cat((sc_1, sc_2), 1)


class SparseGraphSTEncoder(Module):
    """
    GraphST encoder using sparse matrix operations throughout.
    GCN: Z = adj @ (X @ W1), reconstruction: H = adj @ (Z @ W2)
    Contrastive: discriminator on (readout, emb, emb_augmented)
    """
    def __init__(self, in_features, out_features, graph_neigh_sparse, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh_sparse = graph_neigh_sparse  # torch sparse
        self.dropout = dropout

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = SparseAvgReadout()

    def forward(self, feat, feat_a, adj_sparse):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj_sparse, z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj_sparse, h)

        emb = F.relu(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj_sparse, z_a)
        emb_a = F.relu(z_a)

        g = self.read(emb, self.graph_neigh_sparse)
        g = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh_sparse)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


# ── Per-donor GraphST training ────────────────────────────────────────────

def train_graphst_donor(X, spatial, device, dim_output=64, epochs=600,
                        lr=0.001, alpha=10, beta=1, n_neighbors=3, seed=41):
    """
    Train GraphST on a single donor's cells. Returns embeddings [N, dim_output].
    All matrices are sparse to handle large donors (up to 200k cells).
    """
    n_cells = X.shape[0]
    n_genes = X.shape[1]

    # 1. Build sparse graph
    adj_sp, neigh_sp = build_sparse_adj_and_neigh(spatial, n_neighbors=n_neighbors)
    adj_norm = normalize_sparse_adj(adj_sp)

    # Convert to torch sparse on device
    adj_torch = sparse_mx_to_torch(adj_norm).to(device)
    # graph_neigh needs self-loop for readout
    neigh_with_eye = neigh_sp + sp.eye(n_cells)
    neigh_torch = sparse_mx_to_torch(neigh_with_eye.tocsr()).to(device)

    # 2. Features (already preprocessed: log1p)
    feat = torch.FloatTensor(X).to(device)

    # 3. Contrastive labels [N, 2]: [1, 0] for each cell
    label_CSL = torch.FloatTensor(
        np.concatenate([np.ones((n_cells, 1)), np.zeros((n_cells, 1))], axis=1)
    ).to(device)

    # 4. Model
    model = SparseGraphSTEncoder(n_genes, dim_output, neigh_torch).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Train
    model.train()
    for epoch in range(epochs):
        # Augmented features: random permutation
        ids = np.random.permutation(n_cells)
        feat_a = feat[ids]

        hiden, emb, ret, ret_a = model(feat, feat_a, adj_torch)

        loss_sl = loss_fn(ret, label_CSL) + loss_fn(ret_a, label_CSL)
        loss_feat = F.mse_loss(feat, emb)
        loss = alpha * loss_feat + beta * loss_sl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 6. Extract embeddings
    model.eval()
    with torch.no_grad():
        ids = np.random.permutation(n_cells)
        feat_a = feat[ids]
        hiden, emb, _, _ = model(feat, feat_a, adj_torch)
        embeddings = hiden.cpu().numpy()  # [N, dim_output]

    # Cleanup
    del model, feat, feat_a, adj_torch, neigh_torch, label_CSL
    torch.cuda.empty_cache()

    return embeddings


# ── Data loading (same split as rest of benchmark) ─────────────────────────

def load_and_split(npz_path, seed=3028):
    data = np.load(npz_path, allow_pickle=True)
    X = np.log1p(data['X'].astype(np.float32))
    y_c = data['y_class']
    y_sc = data['y_subclass']
    y_st = data['y_supertype']
    batch = data['batch_donor']
    spatial = data['spatial'].astype(np.float32)
    meta = data['meta'].item()

    unique_donors = np.unique(batch)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_donors)

    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]

    return {
        'X': X, 'y_class': y_c, 'y_subclass': y_sc, 'y_supertype': y_st,
        'batch': batch, 'spatial': spatial,
        'train_donors': train_donors, 'val_donors': val_donors,
        'test_donors': test_donors, 'meta': meta,
    }


# ── LR evaluation ─────────────────────────────────────────────────────────

def train_and_eval_lr(X_train, y_train, X_test, y_test, n_classes, task_name, logger):
    logger.info(f"\n--- {task_name} ({n_classes} classes) ---")
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    solver = 'saga' if n_classes > 30 else 'lbfgs'
    clf = LogisticRegression(
        max_iter=1000, solver=solver, multi_class='multinomial',
        C=1.0, n_jobs=-1, verbose=0,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='micro', zero_division=0)

    logger.info(f"[{task_name.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
    logger.info(f"[{task_name.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")

    # Per-class
    per_p, per_r, _, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0)
    parts = [f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
             for i in range(len(per_p)) if support[i] > 0]
    logger.info(f"[{task_name.upper()}] Per-class: {' | '.join(parts)}")

    # AUC-ROC
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(len(y_test)), y_test] = 1
    micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
    valid_aucs = []
    for c in range(n_classes):
        y_bin = (y_test == c).astype(int)
        if len(np.unique(y_bin)) == 2:
            valid_aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    logger.info(f"[{task_name.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{n_classes})")
    logger.info("-" * 50)

    return {'macro_f1': macro_f1, 'micro_f1': micro_f1}


# ── Main ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="GraphST embeddings + LR")
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--dim_output', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--n_neighbors', type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    log_dir = './experiments/GraphST/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_d{args.dim_output}_e{args.epochs}_s{args.seed}.log')
    logger.info(f"Args: {args}")

    # 1. Load data
    logger.info("Loading data...")
    D = load_and_split(args.npz_path, seed=args.seed)
    all_donors = np.unique(D['batch'])
    logger.info(f"Total cells: {len(D['X']):,}, Donors: {len(all_donors)}")
    logger.info(f"Train donors: {sorted(D['train_donors'])}")
    logger.info(f"Val donors:   {sorted(D['val_donors'])}")
    logger.info(f"Test donors:  {sorted(D['test_donors'])}")

    # 2. Train GraphST per-donor, collect embeddings
    logger.info("\n========== GraphST Per-Donor Training ==========")
    embeddings = np.zeros((len(D['X']), args.dim_output), dtype=np.float32)

    for donor_id in sorted(all_donors):
        mask = D['batch'] == donor_id
        n_cells = mask.sum()
        split = ('train' if donor_id in D['train_donors']
                 else 'val' if donor_id in D['val_donors']
                 else 'test')

        logger.info(f"\nDonor {donor_id} ({split}): {n_cells:,} cells")
        t0 = time.time()

        emb = train_graphst_donor(
            X=D['X'][mask],
            spatial=D['spatial'][mask],
            device=device,
            dim_output=args.dim_output,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        embeddings[mask] = emb
        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s, emb shape: {emb.shape}")

    # 3. Save embeddings
    emb_path = os.path.join(log_dir, f'embeddings_d{args.dim_output}_e{args.epochs}_s{args.seed}.npy')
    np.save(emb_path, embeddings)
    logger.info(f"\nEmbeddings saved to {emb_path}")

    # 4. Build train/test masks
    mask_train = np.isin(D['batch'], D['train_donors'])
    mask_val   = np.isin(D['batch'], D['val_donors'])
    mask_test  = np.isin(D['batch'], D['test_donors'])

    X_train_emb = embeddings[mask_train]
    X_test_emb  = embeddings[mask_test]

    logger.info(f"\nTrain embeddings: {X_train_emb.shape}")
    logger.info(f"Test embeddings:  {X_test_emb.shape}")

    # 5. LR classification
    logger.info("\n========== LR Classification on GraphST Embeddings ==========")
    tasks = [
        ('class',     D['meta']['num_class']),
        ('subclass',  D['meta']['num_subclass']),
        ('supertype', D['meta']['num_supertype']),
    ]
    for task_name, n_classes in tasks:
        y_key = f'y_{task_name}'
        train_and_eval_lr(
            X_train_emb, D[y_key][mask_train],
            X_test_emb,  D[y_key][mask_test],
            n_classes, task_name, logger,
        )

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
