#!/usr/bin/env python
"""
scSimilarity N-level for Kukanja datasets.
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random, numpy as np, torch
from src.utils.dataloader_kukanja import build_dataloaders_kukanja
from src.models.fm_nlevel import ScSimilarityNLevel
from src.engines.fm_trainer_kukanja import FMTrainerKukanja
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name='scSimilarity', dropout=0.5)
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--scsi_latent_dim', type=int, default=128)
    parser.add_argument('--scsi_hidden_dim', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--scsi_dropout', type=float, default=0.5)
    parser.add_argument('--scsi_input_dropout', type=float, default=0.4)
    args = parser.parse_args()

    if 'ms' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_MS'
    elif 'eae' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_EAE'
    else:
        dataset_tag = 'Kukanja'

    ft_tag = 'freeze' if args.freeze_backbone else 'full'
    ckpt_tag = f"{ft_tag}_s{args.seed}"
    output_root = os.environ.get('KUKANJA_OUT_ROOT', 'outputs/kukanja_runs')
    log_dir = os.path.join(output_root, dataset_tag)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_scsimilarity_{ckpt_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'scsimilarity_{ckpt_tag}')
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
        args.npz_path, batch_size=args.bs, seed=args.seed, num_workers=args.num_workers)

    args.output_num = info['output_num']
    args.input_dim = info['n_genes']
    args.label_names = info['label_names']
    logger.info(f"Dataset: {args.dataset_tag}")
    logger.info(f"Levels: {list(zip(info['label_names'], info['output_num']))}")

    model = ScSimilarityNLevel(
        n_genes=args.input_dim,
        output_num=info['output_num'],
        latent_dim=args.scsi_latent_dim,
        hidden_dim=args.scsi_hidden_dim,
        dropout=args.scsi_dropout,
        input_dropout=args.scsi_input_dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: scSimilarity N-Level, trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = FMTrainerKukanja(
        model, optimizer, scheduler, device, logger, args,
        label_names=info['label_names'], output_num=info['output_num'])

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
