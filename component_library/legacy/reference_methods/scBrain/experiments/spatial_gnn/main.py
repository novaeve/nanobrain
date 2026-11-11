"""
Spatial GNN (STELLAR-like) experiment for SEA-AD MERFISH cell annotation.

Builds KNN spatial graphs per donor, then trains a GCN with hierarchical heads.

Usage:
    python experiments/spatial_gnn/main.py --device cuda:0 --k_neighbors 15
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.models.spatial_gnn import SpatialGNN, build_knn_graph
from src.engines.spatial_gnn_trainer import SpatialGNNTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def build_donor_subgraphs(npz_path, donor_ids, k=15, seed=3028):
    """Build per-donor subgraphs with KNN spatial edges."""
    data = np.load(npz_path, allow_pickle=True)
    X_all = data['X'].astype(np.float32)
    y_c, y_sc, y_st = data['y_class'], data['y_subclass'], data['y_supertype']
    batch_all = data['batch_donor']
    spatial_all = data['spatial'].astype(np.float32)
    cps_all = data['cps'].astype(np.float32)
    conf_all = data['y_supertype_confidence'].astype(np.float32)

    subgraphs = []
    for did in donor_ids:
        mask = batch_all == did
        sp = spatial_all[mask]
        edge_index = build_knn_graph(sp, k=k)
        subgraphs.append({
            'X': torch.from_numpy(X_all[mask]),
            'y_class': torch.from_numpy(y_c[mask]).long(),
            'y_subclass': torch.from_numpy(y_sc[mask]).long(),
            'y_supertype': torch.from_numpy(y_st[mask]).long(),
            'confidence': torch.from_numpy(conf_all[mask]),
            'spatial': torch.from_numpy(sp),
            'batch_id': torch.from_numpy(batch_all[mask]).long(),
            'cps': torch.from_numpy(cps_all[mask]),
            'edge_index': edge_index,
        })
    return subgraphs


def get_donor_split(npz_path, seed=3028):
    """Same donor-stratified split as the standard pipeline."""
    data = np.load(npz_path, allow_pickle=True)
    batch_all = data['batch_donor']
    unique_donors = np.unique(batch_all)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)
    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]
    return train_donors, val_donors, test_donors


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name='spatial_gnn')
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--k_neighbors', type=int, default=15, help='KNN graph neighbors')
    parser.add_argument('--n_gcn_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--gnn_hidden_dim', type=int, default=256, help='GNN hidden dimension')
    parser.add_argument('--gnn_latent_dim', type=int, default=128, help='GNN latent dimension')
    args = parser.parse_args()

    ckpt_tag = f"k{args.k_neighbors}_gcn{args.n_gcn_layers}_h{args.gnn_hidden_dim}_z{args.gnn_latent_dim}_s{args.seed}"
    log_dir = f'./experiments/spatial_gnn/{args.dataset}/'
    logger = get_logger(log_dir, __name__, f'record_{ckpt_tag}.log')
    logger.info(args)
    return args, log_dir, logger, ckpt_tag


if __name__ == '__main__':
    args, log_dir, logger, ckpt_tag = get_args()
    args.log_dir = log_dir
    args.save_dir = os.path.join(log_dir, 'checkpoints', ckpt_tag)

    set_seed(args.seed)
    device = torch.device(args.device)

    print("====== Spatial GNN ======")

    # 1. Donor split
    train_donors, val_donors, test_donors = get_donor_split(args.npz_path, seed=args.seed)
    logger.info(f"Train donors: {len(train_donors)}, Val: {len(val_donors)}, Test: {len(test_donors)}")

    # 2. Build KNN subgraphs per donor
    logger.info(f"Building KNN graphs (k={args.k_neighbors})...")
    train_sg = build_donor_subgraphs(args.npz_path, train_donors, k=args.k_neighbors)
    val_sg   = build_donor_subgraphs(args.npz_path, val_donors, k=args.k_neighbors)
    test_sg  = build_donor_subgraphs(args.npz_path, test_donors, k=args.k_neighbors)
    logger.info(f"Subgraphs built: train={len(train_sg)}, val={len(val_sg)}, test={len(test_sg)}")

    # 3. Model
    model = SpatialGNN(
        input_dim=args.input_dim,
        hidden_dim=args.gnn_hidden_dim,
        latent_dim=args.gnn_latent_dim,
        n_gcn_layers=args.n_gcn_layers,
        dropout=args.dropout,
        output_num=args.output_num,
    ).to(device)

    # 4. Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    # 5. Trainer
    engine = SpatialGNNTrainerEngine(model, optimizer, scheduler, device, logger, args)

    # 6. Run
    if args.mode == 'train':
        engine.train(train_sg, val_sg)
        engine.test(test_sg)
    else:
        engine.test(test_sg)
