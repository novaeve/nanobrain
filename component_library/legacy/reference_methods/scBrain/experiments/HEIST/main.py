import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.heist.heist_annotation import HEISTForAnnotation
from src.engines.heist_trainer import HEISTTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="HEIST", dropout=0.2)

    parser.add_argument(
        "--npz_path", type=str, default="./data/global_data.npz",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Optional path to pretrained HEIST encoder checkpoint.",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze HEIST encoder; train only cls heads.",
    )
    parser.add_argument("--k_spatial", type=int, default=10,
                        help="k for spatial k-NN graph.")
    parser.add_argument("--k_gene", type=int, default=5,
                        help="k for gene co-expression graph.")
    parser.add_argument("--pe_dim", type=int, default=32,
                        help="Positional encoding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension in graph layers.")
    parser.add_argument("--output_dim", type=int, default=128,
                        help="Encoder output embedding dimension.")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of multi-level graph layers.")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Attention heads in graph layers.")

    args = parser.parse_args()

    ft_tag = "freeze" if args.freeze_backbone else "full"
    ckpt_tag = f"{ft_tag}_s{args.seed}"
    args.ckpt_tag = ckpt_tag

    log_dir = "./experiments/{}/{}/".format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, f"record_{ckpt_tag}.log")
    logger.info(args)

    return args, log_dir, logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args, log_dir, logger = get_args()
    args.log_dir = log_dir
    args.save_dir = os.path.join(log_dir, "checkpoints", args.ckpt_tag)

    set_seed(args.seed)
    device = torch.device(args.device)

    print("====== HEIST ======")

    # 1. Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path,
        batch_size=args.bs,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # 2. Model
    model = HEISTForAnnotation(
        checkpoint_path=args.checkpoint_path,
        output_num=args.output_num,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        k_spatial=args.k_spatial,
        k_gene=args.k_gene,
        pe_dim=args.pe_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    # 3. Optimizer & scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    # 4. Trainer
    engine = HEISTTrainerEngine(model, optimizer, scheduler, device, logger, args)

    # 5. Run
    if args.mode == "train":
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
