import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import numpy as np
import torch

from src.utils.dataloader import build_dataloaders
from src.models.geneformer.geneformer_annotation import GeneformerForAnnotation
from src.engines.scgpt_trainer import ScGPTTrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config
import anndata


def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="Geneformer", dropout=0.2)

    parser.add_argument(
        "--npz_path",
        type=str,
        default="./data/global_data.npz",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default="./external/Geneformer_code_only",
        help="Geneformer code directory (contains config.json and geneformer/).",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./weights/geneformer/"
                "4014c514594e479d8fd84d2c65dd94fc/Research/model.safetensors",
        help="Path to Geneformer 316M model.safetensors.",
    )
    parser.add_argument(
        "--h5ad_path",
        type=str,
        default="./data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad",
        help="Original MERFISH h5ad for gene names.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze BERT backbone; train only cls heads.",
    )

    args = parser.parse_args()

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

    print("====== Geneformer ======")

    # 1. Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path    = args.npz_path,
        batch_size  = args.bs,
        seed        = args.seed,
        num_workers = args.num_workers,
    )

    # 2. Gene names
    adata = anndata.read_h5ad(args.h5ad_path, backed="r")
    gene_names = [g for g in adata.var_names if not g.startswith("Blank")]

    # 3. Model
    model = GeneformerForAnnotation(
        code_dir        = args.code_dir,
        weights_path    = args.weights_path,
        gene_names      = gene_names,
        output_num      = args.output_num,
        dropout         = args.dropout,
        freeze_backbone = args.freeze_backbone,
    ).to(device)

    # 4. Optimizer & scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
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
