#!/usr/bin/env python
"""
Geneformer N-level for Kukanja MS (human) dataset.
Usage:
  python experiments/kukanja/main_geneformer.py \
      --npz_path ./data/kukanja/kukanja_ms.npz \
      --h5ad_path data/raw/kukanja/MS_xenium_data_v5_with_images_tmap.h5ad \
      --device cuda:6 --seed 3028 --mode train [--freeze_backbone]
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random, numpy as np, torch
from src.utils.dataloader_kukanja import build_dataloaders_kukanja
from src.models.fm_nlevel import GeneformerNLevel
from src.engines.fm_trainer_kukanja import FMTrainerKukanja
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="Geneformer", dropout=0.2)
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--code_dir', type=str,
                        default='./external/Geneformer_code_only')
    parser.add_argument('--weights_path', type=str,
                        default='./weights/geneformer/'
                                '4014c514594e479d8fd84d2c65dd94fc/Research/model.safetensors')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Kukanja h5ad for gene names')
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
    logger = get_logger(log_dir, __name__, f'record_geneformer_{ckpt_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'geneformer_{ckpt_tag}')
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

    # Gene names from h5ad
    import anndata
    adata = anndata.read_h5ad(args.h5ad_path, backed='r')
    gene_names = list(adata.var_names)
    logger.info(f"Gene panel: {len(gene_names)} genes")

    model = GeneformerNLevel(
        code_dir=args.code_dir,
        weights_path=args.weights_path,
        gene_names=gene_names,
        output_num=info['output_num'],
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: Geneformer N-Level, trainable params: {n_params:,}")

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
