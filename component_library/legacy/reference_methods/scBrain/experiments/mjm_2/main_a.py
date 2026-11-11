#!/usr/bin/env python
"""
scBrainDepth entry point: Laminar Depth Injection.
Stage 1: train DepthImputer (if not already done)
Stage 2: train scBrain with depth-injected hierarchical classification
"""
import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader_v2 import build_dataloaders_v2
from src.models.scbrain import scBrainDepth
from src.engines.trainer_2a import TrainerEngine_2a
from experiments.mjm_2.train_depth_imputer import DepthImputer
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.add_argument('--npz_path',       type=str,   default='./data/global_data_v2.npz')
    parser.add_argument('--imputer_path',   type=str,   default='./data/depth_imputer.pth')
    parser.add_argument('--depth_mlp_dim',  type=int,   default=64)
    parser.add_argument('--lambda_depth',   type=float, default=0.1)
    args = parser.parse_args()
    args.model_name = 'scBrainDepth'

    ckpt_tag = (
        f"e{args.e_layers}_d{args.d_layers}"
        f"_enc{args.enc_hidden_dim}_dec{args.dec_hidden_dim}"
        f"_z{args.latent_dim}_s{args.seed}"
        f"_res{args.residual_mode}"
        f"_ldepth{args.lambda_depth}"
    )
    log_dir = './experiments/mjm_2/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_a_{ckpt_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'a_{ckpt_tag}')
    return args, log_dir, logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args, log_dir, logger = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders_v2(
        args.npz_path, batch_size=args.bs, seed=args.seed, num_workers=args.num_workers
    )

    # ── Load DepthImputer (must be pre-trained) ──────────────────────────────
    assert os.path.exists(args.imputer_path), \
        f"DepthImputer not found at {args.imputer_path}. Run train_depth_imputer.py first."
    depth_imputer = DepthImputer(input_dim=args.input_dim, hidden=128).to(device)
    depth_imputer.load_state_dict(torch.load(args.imputer_path, map_location=device))
    depth_imputer.eval()
    logger.info(f'Loaded DepthImputer from {args.imputer_path}')

    # ── Model ────────────────────────────────────────────────────────────────
    model = scBrainDepth(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        enc_hidden_dim=args.enc_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        depth_mlp_dim=args.depth_mlp_dim,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        output_num=args.output_num,
        residual_mode=args.residual_mode,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = TrainerEngine_2a(
        model, optimizer, scheduler, device, logger, args,
        depth_imputer=depth_imputer,
        lambda_depth=args.lambda_depth,
    )

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
