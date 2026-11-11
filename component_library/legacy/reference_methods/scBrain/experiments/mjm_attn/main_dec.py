"""
MJM_Attn_Dec: Transformer encoder + nonlinear decoder heads.

Usage:
    python experiments/mjm_attn/main_dec.py --device cuda:2 --bs 4096
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.mjm_attn import mjm_attn_dec
from src.engines.trainer_attn import TrainerEngine_attn
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
    parser.set_defaults(bs=2048)
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--n_heads', type=int, default=8)
    args = parser.parse_args()
    args.model_name = 'mjm_attn_dec'

    log_tag = f"e{args.e_layers}_d{args.d_layers}_enc{args.enc_hidden_dim}_dec{args.dec_hidden_dim}_z{args.latent_dim}_h{args.n_heads}_bs{args.bs}_s{args.seed}"
    log_dir = f'./experiments/mjm_attn/{args.dataset}/'
    logger = get_logger(log_dir, __name__, f'record_dec_{log_tag}.log')
    logger.info(args)
    args.save_dir = os.path.join(log_dir, 'checkpoints', f'dec_{log_tag}')
    return args, log_dir, logger


if __name__ == '__main__':
    args, log_dir, logger = get_args()
    args.log_dir = log_dir

    set_seed(args.seed)
    device = torch.device(args.device)

    print("====== MJM_ATTN_DEC ======")

    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path, batch_size=args.bs,
        seed=args.seed, num_workers=args.num_workers,
    )

    model = mjm_attn_dec(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        enc_hidden_dim=args.enc_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        n_heads=args.n_heads,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        output_num=args.output_num,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = TrainerEngine_attn(model, optimizer, scheduler, device, logger, args)

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
