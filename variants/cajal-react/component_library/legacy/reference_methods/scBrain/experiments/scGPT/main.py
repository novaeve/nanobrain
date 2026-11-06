import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.scGPT.scGPT_annotation import scGPTForAnnotation
from src.engines.scgpt_trainer import ScGPTTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config


VARIANT_CHECKPOINTS = {
    "brain": "./weights/scgpt/brain",
    "human": "./weights/scgpt/human",
}


def get_args():
    parser = get_public_config()
    # override defaults for scGPT
    parser.set_defaults(model_name="scGPT", dropout=0.2)

    parser.add_argument(
        "--npz_path",
        type=str,
        default="./data/global_data.npz",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="brain",
        choices=["brain", "human"],
        help="scGPT pre-training variant (brain or human).",
    )
    parser.add_argument(
        "--gene_ids_path",
        type=str,
        default="./src/models/scGPT/merfish_indices.npy",
        help="Path to merfish_indices.npy (140 vocab indices).",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the transformer backbone; train only the cls heads.",
    )

    args = parser.parse_args()

    # Set checkpoint_dir and model_name based on variant
    args.checkpoint_dir = VARIANT_CHECKPOINTS[args.variant]
    args.model_name     = f"scGPT_{args.variant}"

    ft_tag   = "freeze" if args.freeze_backbone else "full"
    ckpt_tag = f"{ft_tag}_s{args.seed}"
    args.ckpt_tag = ckpt_tag

    log_dir = "./experiments/{}/{}/".format(args.model_name, args.dataset)
    logger  = get_logger(log_dir, __name__, f"record_{ckpt_tag}.log")
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
    args.log_dir  = log_dir
    args.save_dir = os.path.join(log_dir, "checkpoints", args.ckpt_tag)

    set_seed(args.seed)
    device = torch.device(args.device)

    print("====== scGPT ======")

    # 1. Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path    = args.npz_path,
        batch_size  = args.bs,
        seed        = args.seed,
        num_workers = args.num_workers,
    )

    # 2. Model
    gene_ids = torch.from_numpy(np.load(args.gene_ids_path)).long()
    model = scGPTForAnnotation(
        checkpoint_dir  = args.checkpoint_dir,
        gene_ids        = gene_ids,
        output_num      = args.output_num,
        n_bins          = 51,
        dropout         = args.dropout,
        freeze_backbone = args.freeze_backbone,
    ).to(device)

    # 3. Optimizer & scheduler (only trainable params)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    # 4. Trainer
    engine = ScGPTTrainerEngine(model, optimizer, scheduler, device, logger, args)

    # 5. Run
    if args.mode == "train":
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
