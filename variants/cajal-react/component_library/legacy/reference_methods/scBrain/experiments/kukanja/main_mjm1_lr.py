#!/usr/bin/env python
"""
MJM_1_LR N-level entry point for Kukanja datasets.
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader_kukanja import build_dataloaders_kukanja
from src.models.mjm_nlevel import mjm_1_lr_nlevel
from src.engines.trainer_kukanja import KukanjaTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to packed Kukanja NPZ file')
    args = parser.parse_args()
    args.model_name = 'mjm_1_lr_nlevel'

    if 'ms' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_MS'
    elif 'eae' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_EAE'
    else:
        dataset_tag = 'Kukanja'

    ckpt_tag = (
        f"e{args.e_layers}"
        f"_enc{args.enc_hidden_dim}"
        f"_z{args.latent_dim}_s{args.seed}"
    )
    output_root = os.environ.get('KUKANJA_OUT_ROOT', 'outputs/kukanja_runs')
    log_dir = os.path.join(output_root, dataset_tag)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_mjm1_lr_{ckpt_tag}.log')
    logger.info(f"Args: {args}")
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'mjm1_lr_{ckpt_tag}')
    args.dataset_tag = dataset_tag
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

    train_loader, val_loader, test_loader, info = build_dataloaders_kukanja(
        args.npz_path, batch_size=args.bs, seed=args.seed, num_workers=args.num_workers
    )

    args.output_num = info['output_num']
    args.input_dim = info['n_genes']
    args.label_names = info['label_names']

    logger.info(f"Dataset: {args.dataset_tag}")
    logger.info(f"  Cells: train={info['n_train']}, val={info['n_val']}, test={info['n_test']}")
    logger.info(f"  Genes: {info['n_genes']}")
    logger.info(f"  Levels: {list(zip(info['label_names'], info['output_num']))}")

    model = mjm_1_lr_nlevel(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        e_layers=args.e_layers,
        enc_hidden_dim=args.enc_hidden_dim,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        output_num=args.output_num,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model_name}, params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = KukanjaTrainerEngine(
        model, optimizer, scheduler, device, logger, args,
        label_names=info['label_names'],
        output_num=info['output_num'],
    )

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
