#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random, json
import numpy as np
import torch
import anndata

from src.utils.dataloader_kukanja import build_dataloaders_kukanja
from src.models.fm_nlevel import NicheformerNLevel
from src.engines.fm_trainer_kukanja import FMTrainerKukanja
from src.utils.logger import get_logger
from src.utils.args import get_public_config

NICHEFORMER_ASSET_ROOT = (
    "./weights/nicheformer"
)


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="Nicheformer", dropout=0.2)
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--h5ad_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default=f'{NICHEFORMER_ASSET_ROOT}/nicheformer.ckpt')
    parser.add_argument('--vocab_path', type=str, default=f'{NICHEFORMER_ASSET_ROOT}/model.h5ad')
    parser.add_argument('--merfish_mean_path', type=str, default=f'{NICHEFORMER_ASSET_ROOT}/merfish_mean_script.npy')
    parser.add_argument('--gene_name_to_ens_path', type=str, default=f'{NICHEFORMER_ASSET_ROOT}/gene_name_id_dict_gc104M.pkl')
    parser.add_argument('--freeze_backbone', action='store_true')
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
    logger = get_logger(log_dir, __name__, f'record_nicheformer_{ckpt_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'nicheformer_{ckpt_tag}')
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
    logger.info(f"Dataset: {args.dataset_tag}, Levels: {list(zip(info['label_names'], info['output_num']))}")

    adata = anndata.read_h5ad(args.h5ad_path, backed='r')
    gene_names = list(adata.var_names)
    logger.info(f"Gene panel: {len(gene_names)} genes")

    model = NicheformerNLevel(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        merfish_mean_path=args.merfish_mean_path,
        gene_name_to_ens_path=args.gene_name_to_ens_path,
        gene_names=gene_names,
        output_num=info['output_num'],
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: Nicheformer N-Level, trainable params: {n_params:,}")

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
