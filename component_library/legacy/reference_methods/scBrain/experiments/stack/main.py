import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import json
import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.stack_model.stack_annotation import StackForAnnotation
from src.engines.scgpt_trainer import ScGPTTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config

STACK_ASSET_ROOT = "./weights/stack"


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="Stack", dropout=0.2)

    parser.add_argument(
        "--npz_path",
        type=str,
        default="./data/global_data.npz",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=f"{STACK_ASSET_ROOT}/bc_large.ckpt",
        help="Path to Stack Lightning checkpoint.",
    )
    parser.add_argument(
        "--gene_list_path",
        type=str,
        default=f"{STACK_ASSET_ROOT}/basecount_1000per_15000max.pkl",
        help="Path to Stack gene vocabulary pickle.",
    )
    parser.add_argument(
        "--gene_names_path",
        type=str,
        default="./data/merfish_gene_names.json",
        help="JSON file with 140 MERFISH gene names.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze Stack backbone; train only cls heads.",
    )

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

    print("====== Stack ======")

    # 1. Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path,
        batch_size=args.bs,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # 2. Gene names
    with open(args.gene_names_path) as f:
        gene_names = json.load(f)

    # 3. Model
    model = StackForAnnotation(
        checkpoint_path=args.checkpoint_path,
        gene_list_path=args.gene_list_path,
        gene_names=gene_names,
        output_num=args.output_num,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # 4. Optimizer & scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    # 5. Trainer
    engine = ScGPTTrainerEngine(model, optimizer, scheduler, device, logger, args)

    # 6. Run
    if args.mode == "train":
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
