#!/usr/bin/env python
"""
scGPT N-level for Kukanja (brain variant for EAE mouse, human for MS).
Usage:
  python experiments/kukanja/main_scgpt.py \
      --npz_path ./data/kukanja/kukanja_eae.npz \
      --h5ad_path data/raw/kukanja/EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad \
      --ckpt_dir ./weights/scgpt/brain \
      --device cuda:6 --seed 3028 --mode train [--freeze_backbone]
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random, numpy as np, torch
from src.utils.dataloader_kukanja import build_dataloaders_kukanja
from src.models.fm_nlevel import ScGPTNLevel
from src.engines.fm_trainer_kukanja import FMTrainerKukanja
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="scGPT", dropout=0.2)
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to scGPT checkpoint directory')
    parser.add_argument('--h5ad_path', type=str, required=True)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--variant', type=str, default='brain',
                        choices=['brain', 'human'])
    args = parser.parse_args()

    if 'ms' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_MS'
    elif 'eae' in os.path.basename(args.npz_path).lower():
        dataset_tag = 'Kukanja_EAE'
    else:
        dataset_tag = 'Kukanja'

    ft_tag = 'freeze' if args.freeze_backbone else 'full'
    ckpt_tag = f"{ft_tag}_{args.variant}_s{args.seed}"
    output_root = os.environ.get('KUKANJA_OUT_ROOT', 'outputs/kukanja_runs')
    log_dir = os.path.join(output_root, dataset_tag)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_scgpt_{ckpt_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'scgpt_{ckpt_tag}')
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
    args.label_names = info['label_names']
    logger.info(f"Dataset: {args.dataset_tag}, Variant: {args.variant}")
    logger.info(f"Levels: {list(zip(info['label_names'], info['output_num']))}")

    import anndata
    adata = anndata.read_h5ad(args.h5ad_path, backed='r')
    gene_names = list(adata.var_names)
    logger.info(f"Gene panel: {len(gene_names)} genes")

    model = ScGPTNLevel(
        ckpt_dir=args.ckpt_dir,
        gene_names=gene_names,
        output_num=info['output_num'],
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: scGPT N-Level ({args.variant}), trainable params: {n_params:,}")

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
