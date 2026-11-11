import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """兼容 PyTorch < 2.4 的 RMSNorm 实现"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, expansion_factor=2.67, bias=True, dropout=0.3):
        super(SwiGLU, self).__init__()
        hidden_features = int(out_features * expansion_factor)
        self.W1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.W2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.W3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        x = self.W3(x)
        return x

class FFN(nn.Module):
    def __init__(self, n_layers=3, model_dim=256, expansion_factor=2.67, bias=True, dropout=0.3):
        super(FFN, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            SwiGLU(model_dim, model_dim, expansion_factor, bias, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(model_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x) + x)
        return x

class Hierarchical_Decoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, output_num=[3, 24, 137], 
                 is_hierarchical=True, dropout=0.1, apply_softmax=False):
        """
        Args:
            input_dim: Encoder 传过来的隐向量维度 (Latent dim)
            hidden_dim: Decoder 内部 MLPs 的隐藏层维度
            output_num: 分类树各层级的类别数量列表
            is_hierarchical: 开关。True 为串联分层+残差，False 为完全并列
            apply_softmax: 是否在最后一层应用 Softmax 
                          (注: 若使用 nn.CrossEntropyLoss，此处应保持 False 输出 Logits)
        """
        super(Hierarchical_Decoder, self).__init__()
        self.output_num = output_num
        self.is_hierarchical = is_hierarchical
        self.apply_softmax = apply_softmax
        
        # 1. 维度投影层：将输入的隐向量对齐到隐藏层维度
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # 2. 核心网络：N 个 MLP 块组成的 ModuleList
        self.decoders = nn.ModuleList([
            SwiGLU(hidden_dim, hidden_dim, dropout=dropout) for _ in range(len(output_num))
        ])
        
        # 3. 分类头：N 个 Linear 层组成的 ModuleList
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, out_classes) for out_classes in output_num
        ])

    def forward(self, x):
        # 投影到一致的 hidden_dim
        x = self.input_proj(x)
        
        outputs = []
        # 初始化当前特征为原始输入
        curr_feat = x 
        
        for i in range(len(self.output_num)):
            if self.is_hierarchical:
                # ---------------------------------------------------------
                # 【分层模式 (Hierarchical + Residual)】
                # 当前特征进入第 i 层的 MLP
                mlp_out = self.decoders[i](curr_feat)
                
                # 核心设计：残差连接 (Residual)
                # 新特征 = 提取的层级特征 + 原始/上一层特征
                # 这样既有分层的深度概念，又并行保留了原始信息
                curr_feat = curr_feat + mlp_out 
                
                # 将融合后的特征输入到分类头
                head_input = curr_feat
                # ---------------------------------------------------------
            else:
                # ---------------------------------------------------------
                # 【完全并行模式 (Flat / Parallel)】
                # 所有的 MLP 都只看最初始的投影输入 x，互不干扰
                mlp_out = self.decoders[i](x)
                head_input = mlp_out
                # ---------------------------------------------------------
            
            # 分类头输出 (Logits)
            logits = self.heads[i](head_input)
            
            # 根据需求决定是否加 Softmax
            if self.apply_softmax:
                logits = torch.softmax(logits, dim=-1)
                
            outputs.append(logits)
            
        return outputs



class mjm(nn.Module):
    def __init__(self, 
                 input_dim=180, 
                 latent_dim=20, 
                 e_layers=3, 
                 d_layers=1, 
                 enc_hidden_dim=256, 
                 dec_hidden_dim=256, 
                 expansion_factor=2.67, 
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 is_hierarchical=True,
                 ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim, expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        self.recon_decoder = SwiGLU(latent_dim, input_dim)

        self.decoder = Hierarchical_Decoder(
            input_dim=latent_dim,
            hidden_dim=dec_hidden_dim,
            output_num=output_num,
            is_hierarchical=is_hierarchical,
            dropout=dropout,
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.recon_decoder(z)
        class_outputs = self.decoder(z)
        return recon, class_outputs, z


# ============================================================
# mjm_1: 三级级联编解码器 + 跨层残差分类
# ============================================================

# 内部工具函数：构建一个「线性投影 + FFN」解码块
def _make_dec_block(in_dim: int, out_dim: int, n_layers: int,
                    expansion_factor: float, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        FFN(n_layers=n_layers, model_dim=out_dim,
            expansion_factor=expansion_factor, dropout=dropout),
    )


class mjm_1(nn.Module):
    """
    mjm_1：三级级联编解码器，层间残差分类

    架构示意
    ─────────────────────────────────────────────────────────────
    X [B, input_dim]
    │
    ├─ E1 (Linear + FFN×e_layers) ──────────────────────────────┐
    │  h1 [B, enc_hidden_dim]                                    D1 (Linear + FFN×d_layers)
    ├─ E2 (FFN×e_layers) ───────────────────────────────────┐   C1 = D1(h1)  → logits1
    │  h2 [B, enc_hidden_dim]                                D2
    └─ E3 (FFN×e_layers + Linear) ─> H [B, latent_dim]      C2 = D2(h2) ⊕ residual(C1/logits1)
       │                                                         → logits2
       ├─ recon_decoder (SwiGLU) ─> x_hat                   D3
                                     C3 = D3(H) ⊕ residual(C2/logits2)
                                         → logits3
    ─────────────────────────────────────────────────────────────

    残差模式 (residual_mode)：
        'feature' : C_{i+1} = D_{i+1}(...) + C_i
                    在 Decoder 隐层特征空间直接残差（维度恒为 dec_hidden_dim）
        'logit'   : C_{i+1} = D_{i+1}(...) + proj(logits_i)
                    将上一级 logits 投影回 dec_hidden_dim 后残差；
                    上级预测信号显式注入下级，信息约束更强
        'none'    : 无残差，三路 Decoder 完全独立

    参数说明：
        input_dim:        输入基因表达维度
        latent_dim:       隐空间维度（E3 输出）
        e_layers:         每个 Encoder FFN 块的层数（Encoder 宜重）
        d_layers:         每个 Decoder FFN 块的层数（Decoder 宜轻）
        enc_hidden_dim:   E1/E2 的隐藏层维度（建议 ≥ dec_hidden_dim）
        dec_hidden_dim:   D1/D2/D3 的输出特征维度（C 的维度）
        expansion_factor: FFN SwiGLU 扩张比
        dropout:          Dropout 概率
        output_num:       三级类别数 [n1, n2, n3]
                          顺序：[C1(E1出口), C2(E2出口), C3(E3/H出口)]
                          示例：[3, 24, 137] → Class / Subclass / Supertype
        residual_mode:    层间残差方式，见上方说明

    返回 (forward)：
        recon:         重构基因表达     [B, input_dim]
        class_outputs: 三级 logits 列表 [[B,n1], [B,n2], [B,n3]]
        H:             隐向量           [B, latent_dim]
    """
    def __init__(self,
                 input_dim=180,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 residual_mode='feature',
                 spatial_dim=0,
                 ):
        super().__init__()
        assert len(output_num) == 3, "output_num 须含 3 个元素，对应 C1/C2/C3"
        assert residual_mode in ('feature', 'logit', 'none'), \
            "residual_mode 须为 'feature' | 'logit' | 'none'"
        self.residual_mode = residual_mode
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim

        # ── Encoders（重）：逐级压缩基因特征 ──────────────────────────────────
        self.E1 = nn.Sequential(
            nn.Linear(input_dim + spatial_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
        )
        self.E2 = FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                      expansion_factor=expansion_factor, dropout=dropout)
        self.E3 = nn.Sequential(
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        # ── Decoders（轻）：各层级特征解码 ───────────────────────────────────
        self.D1 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D2 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D3 = _make_dec_block(latent_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)

        # ── 残差投影（仅 'logit' 模式）──────────────────────────────────────
        if residual_mode == 'logit':
            self.proj_c1_to_c2 = nn.Linear(output_num[0], dec_hidden_dim)
            self.proj_c2_to_c3 = nn.Linear(output_num[1], dec_hidden_dim)

        # ── Classification heads ────────────────────────────────────────────
        self.head1 = nn.Linear(dec_hidden_dim, output_num[0])
        self.head2 = nn.Linear(dec_hidden_dim, output_num[1])
        self.head3 = nn.Linear(dec_hidden_dim, output_num[2])

        # ── Reconstruction decoder ──────────────────────────────────────────
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

    def forward(self, x):
        # ── 编码 ─────────────────────────────────────────────────────────────
        h1 = self.E1(x)   # [B, enc_hidden_dim]
        h2 = self.E2(h1)  # [B, enc_hidden_dim]
        H  = self.E3(h2)  # [B, latent_dim]

        # ── 解码 + 分类（含层间残差）─────────────────────────────────────────
        C1      = self.D1(h1)     # [B, dec_hidden_dim]
        logits1 = self.head1(C1)  # [B, n1]

        if self.residual_mode == 'feature':
            C2 = self.D2(h2) + C1
        elif self.residual_mode == 'logit':
            C2 = self.D2(h2) + self.proj_c1_to_c2(logits1)
        else:
            C2 = self.D2(h2)

        logits2 = self.head2(C2)  # [B, n2]

        if self.residual_mode == 'feature':
            C3 = self.D3(H) + C2
        elif self.residual_mode == 'logit':
            C3 = self.D3(H) + self.proj_c2_to_c3(logits2)
        else:
            C3 = self.D3(H)

        logits3 = self.head3(C3)  # [B, n3]

        # ── 重构 ─────────────────────────────────────────────────────────────
        recon = self.recon_decoder(H)  # [B, input_dim]

        return recon, [logits1, logits2, logits3], H
