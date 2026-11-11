"""
MJM_LR experiment: mjm/mjm_1 with LR classification heads.

Usage:
    python experiments/mjm_lr/main.py --model_name mjm_lr --device cuda:0
    python experiments/mjm_lr/main.py --model_name mjm_1_lr --device cuda:0
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.mjm_lr import mjm_lr, mjm_1_lr
from src.engines.trainer import TrainerEngine
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

    log_tag = f"e{args.e_layers}_enc{args.enc_hidden_dim}_z{args.latent_dim}_s{args.seed}"
    log_dir = f'./experiments/mjm_lr/{args.dataset}/'
    logger = get_logger(log_dir, __name__, f'record_{args.model_name}_{log_tag}.log')
    logger.info(args)
    return args, log_dir, logger, log_tag


if __name__ == '__main__':
    args, log_dir, logger, log_tag = get_args()
    args.log_dir = log_dir
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'{args.model_name}_{log_tag}')

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"====== {args.model_name.upper()} ======")

    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path, batch_size=args.bs,
        seed=args.seed, num_workers=args.num_workers,
    )

    if args.model_name == 'mjm_lr':
        model = mjm_lr(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            e_layers=args.e_layers,
            enc_hidden_dim=args.enc_hidden_dim,
            expansion_factor=args.expansion_factor,
            dropout=args.dropout,
            output_num=args.output_num,
        ).to(device)
    elif args.model_name == 'mjm_1_lr':
        model = mjm_1_lr(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            e_layers=args.e_layers,
            enc_hidden_dim=args.enc_hidden_dim,
            expansion_factor=args.expansion_factor,
            dropout=args.dropout,
            output_num=args.output_num,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model_name}, use 'mjm_lr' or 'mjm_1_lr'")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = TrainerEngine(model, optimizer, scheduler, device, logger, args)

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
