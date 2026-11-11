"""
MJM MOO experiment: multi-objective optimization for mjm_1 / mjm_1_lr.

Usage:
    # Phase 1: MOO comparison
    python experiments/mjm_moo/main.py --model_name mjm_1_lr --moo_method uncertainty --device cuda:0
    python experiments/mjm_moo/main.py --model_name mjm_1    --moo_method pcgrad      --device cuda:1

    # Phase 2: deeper encoder
    python experiments/mjm_moo/main.py --model_name mjm_1_lr --moo_method uncertainty --e_layers 5 --device cuda:2

    # Phase 3: with spatial
    python experiments/mjm_moo/main.py --model_name mjm_1_lr --moo_method uncertainty --spatial_dim 2 --device cuda:3
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.mjm import mjm_1
from src.models.mjm_lr import mjm_1_lr
from src.engines.trainer_moo import TrainerEngine_MOO
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_args():
    parser = get_public_config()
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    args = parser.parse_args()

    sp_tag = '_spatial' if args.spatial_dim > 0 else ''
    log_tag = f"{args.model_name}_{args.moo_method}_{args.e_layers}l_s{args.seed}{sp_tag}"
    args.ckpt_tag = log_tag

    log_dir = f'./experiments/mjm_moo/{args.dataset}/'
    logger = get_logger(log_dir, __name__, f'record_{log_tag}.log')
    logger.info(args)
    return args, log_dir, logger, log_tag


if __name__ == '__main__':
    args, log_dir, logger, log_tag = get_args()
    args.log_dir = log_dir
    args.save_dir = os.path.join(log_dir, 'checkpoints', log_tag)

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"====== {args.model_name} + {args.moo_method} ======")

    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path, batch_size=args.bs,
        seed=args.seed, num_workers=args.num_workers,
    )

    model_kwargs = dict(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        e_layers=args.e_layers,
        enc_hidden_dim=args.enc_hidden_dim,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        output_num=args.output_num,
        spatial_dim=args.spatial_dim,
    )

    if args.model_name == 'mjm_1_lr':
        model = mjm_1_lr(**model_kwargs).to(device)
    elif args.model_name == 'mjm_1':
        model_kwargs['d_layers'] = args.d_layers
        model_kwargs['dec_hidden_dim'] = args.dec_hidden_dim
        model_kwargs['residual_mode'] = args.residual_mode
        model = mjm_1(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = TrainerEngine_MOO(model, optimizer, scheduler, device, logger, args)

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
