import argparse

def get_public_config():
    parser = argparse.ArgumentParser(description="scBrain configuration")
    
    # ==========================================
    # 1. 基础环境与路径配置 (Base & Path Settings)
    # ==========================================
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='SEA_AD_MTP_ST')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--model_name', type=str, default='scBrain',
                        help="Public model name. Use scBrain for the residual hierarchical decoder; legacy mjm_* names are accepted for reproduction.")
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--mode', type=str, default='train')
    
    parser.add_argument('--num_workers', type=int, default=8)

    # ==========================================
    # 2. 训练与优化配置 (Optimization Settings)
    # ==========================================
    # 整合 --bs 和 --batch_size，统一命名为 batch_size，以第一次的值 64 为准
    parser.add_argument('--bs', type=int, default=512)
    # 整合 --max_epochs 和 --epochs，统一命名为 max_epochs
    parser.add_argument('--max_epochs', type=int, default=500) 
    parser.add_argument('--patience', type=int, default=30) # 合并了 30 和 10，以第一次的 30 为准
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ==========================================
    # 3. 模型架构配置 (Model Settings)
    # ==========================================
    # 合并了两次出现的 input_dim，严格遵守你的要求，以第一次的值 3 为准 (见下方避坑提醒)
    parser.add_argument('--input_dim', type=int, default=140)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--enc_hidden_dim', type=int, default=256)
    parser.add_argument('--dec_hidden_dim', type=int, default=256)
    parser.add_argument('--expansion_factor', type=float, default=2.67)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 使用 nargs='+' 接收多个整数输入，例如在命令行输入: --output_num 3 24 137
    parser.add_argument('--output_num', type=int, nargs='+', default=[3, 24, 137], help='List of output dimensions for hierarchical classification')
    
    # 默认是 True，所以使用 store_false。
    # 命令行如果不加此参数，args.is_hierarchical 就是 True；如果加上 --no_hierarchical，它就会变成 False
    parser.add_argument('--no_hierarchical', action='store_false', dest='is_hierarchical', help='Disable hierarchical processing (defaults to True)')

    # ==========================================
    # 4. scBrain residual decoder settings
    # ==========================================
    parser.add_argument('--residual_mode', type=str, default='feature',
                        choices=['feature', 'logit', 'none'],
                        help=(
                            "层间残差连接方式（scBrain 使用）:\n"
                            "  'feature' : C_{i+1} = D_{i+1}(...) + C_i，在隐层特征空间做残差（默认）\n"
                            "  'logit'   : C_{i+1} = D_{i+1}(...) + proj(logits_i)，logit 投影残差\n"
                            "  'none'    : 三路 Decoder 完全独立，无残差"
                        ))

    # ==========================================
    # 5. 多目标优化配置 (MOO Settings)
    # ==========================================
    parser.add_argument('--moo_method', type=str, default='equal',
                        choices=['equal', 'uncertainty', 'pcgrad', 'gradnorm'],
                        help='Multi-objective optimization strategy')
    parser.add_argument('--alpha_gradnorm', type=float, default=1.5,
                        help='GradNorm asymmetry parameter (higher = more aggressive balancing)')
    parser.add_argument('--spatial_dim', type=int, default=0,
                        help='Spatial coordinate dimensions to append (0=none, 2=xy)')

    return parser
