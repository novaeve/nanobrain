import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import random
import numpy as np
import torch
from src.utils.dataloader import build_dataloaders
from src.models.scbrain import scBrain, scBrainFlat
from src.engines.trainer import TrainerEngine
from src.utils.logger import get_logger
from src.utils.args import get_public_config


def canonical_model_name(name: str) -> str:
    aliases = {
        "scbrain": "scBrain",
        "mjm_1": "scBrain",
        "mjm": "scBrainFlat",
        "scbrainflat": "scBrainFlat",
    }
    return aliases.get(name.lower(), name)



def get_args():
    parser = get_public_config()
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    args = parser.parse_args()
    args.model_name = canonical_model_name(args.model_name)

    # 用相同规则构造 log 文件名（与 ckpt_tag 一致，避免多实验互相覆盖）
    log_tag = (
        f"e{args.e_layers}_d{args.d_layers}"
        f"_enc{args.enc_hidden_dim}_dec{args.dec_hidden_dim}"
        f"_z{args.latent_dim}_s{args.seed}"
    )
    if args.model_name == 'scBrain':
        log_tag += f"_res{args.residual_mode}"

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, f'record_{log_tag}.log')
    logger.info(args)

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
    args.log_dir = log_dir

    # ── Checkpoint 路径：嵌入关键超参，方便对比实验 ──────────────────────────
    # 公共标签：所有模型共用
    ckpt_tag = (
        f"e{args.e_layers}_d{args.d_layers}"
        f"_enc{args.enc_hidden_dim}_dec{args.dec_hidden_dim}"
        f"_z{args.latent_dim}"
        f"_s{args.seed}"
    )
    # scBrain 追加 residual_mode
    if args.model_name == 'scBrain':
        ckpt_tag += f"_res{args.residual_mode}"

    args.save_dir = os.path.join(log_dir, 'checkpoints', ckpt_tag)
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    print("====== scBrain ======")
    
    # 1. Dataloader 确定
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path, 
        batch_size=args.bs, 
        seed=args.seed, 
        num_workers=args.num_workers
    )
    
    # 2. 模型调用
    if args.model_name == 'scBrainFlat':
        model = scBrainFlat(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            enc_hidden_dim=args.enc_hidden_dim,
            dec_hidden_dim=args.dec_hidden_dim,
            expansion_factor=args.expansion_factor,
            dropout=args.dropout,
            output_num=args.output_num,
            is_hierarchical=args.is_hierarchical,
        ).to(device)
    elif args.model_name == 'scBrain':
        model = scBrain(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            enc_hidden_dim=args.enc_hidden_dim,
            dec_hidden_dim=args.dec_hidden_dim,
            expansion_factor=args.expansion_factor,
            dropout=args.dropout,
            output_num=args.output_num,
            residual_mode=args.residual_mode,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}. Use 'scBrain' or legacy aliases 'mjm_1'/'mjm'.")
    
    # 3. 优化器、Schedule 确定
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    
    # 4. 传到 Train Engine 里
    engine = TrainerEngine(model, optimizer, scheduler, device, logger, args)
    
    # 5. 执行 Train 和 Eval(test)
    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)
