import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from torch.utils.checkpoint import checkpoint

class RowBiMamba(nn.Module):
    def __init__(self, d_model=512, d_state=64, grid_size=174):
        super().__init__()
        # 前向和反向 SSM 作为网络的主力模块，负责建模水平方向的共表达依赖。
        # chunk的边界是SSM的记忆断点。为了让行内任意两个基因之间有完整的双向信息流而不被截断，
        # 将一整行作为一个 chunk，因此 chunk_size 设置为网格的宽度(行长度)。
        self.forward_ssm  = Mamba2(
            d_model    = d_model,
            d_state    = d_state,
            d_conv     = 4,
            chunk_size = grid_size,    # 这里建议使用 174 ，即一整行作为一个 chunk
            headdim    = 64,           # 多头 SSM，512/64=8 个头
        )
        self.backward_ssm = Mamba2(
            d_model    = d_model,
            d_state    = d_state,
            d_conv     = 4,
            chunk_size = grid_size,
            headdim    = 64,
        )
        # 将双方向的信息在最后的维度拼接并利用非线性 MLP 融合交互特征
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B, H, W, D) 
        B, H, W, D = x.shape
        
        # 边界防范：Mamba2 的 chunk_size 必须能被该列宽 W 严格整除
        assert W % self.forward_ssm.chunk_size == 0, f"RowBiMamba 报错: chunk_size {self.forward_ssm.chunk_size} 无法整除网格宽 W {W}"
        
        # 把每一行特征当作独立的序列进行处理，总共 B*H 个序列，每个长度为 W
        x_row = x.view(B * H, W, D)        # (B*H, W, D)
        
        # 前向扫描
        fwd = self.forward_ssm(x_row)      # (B*H, W, D)
        # 反向扫描 (先flip以逆序扫描，之后再flip回来)
        bwd = self.backward_ssm(x_row.flip(1)).flip(1)
        
        # 在特征最后维度拼接并通过线性层投影融合，输出为同维度
        out = self.fusion(torch.cat([fwd, bwd], dim=-1)) # (B*H, W, D)
        
        # 恢复回二维网格形状
        return out.view(B, H, W, D)


class ColSSM(nn.Module):
    def __init__(self, d_model=512, d_state=16, grid_size=174):
        super().__init__()
        # 双向轻量 SSM ：防止因单向扫描导致矩阵上半部分的基因由于扫描顺序而无法接收下半部分流匹配推断信号的问题。
        # 上下两个方向各自建模，保证每个 mask 基因具备充分的垂直领域感受野。
        self.ssm_fwd = Mamba2(
            d_model    = d_model,
            d_state    = d_state,
            d_conv     = 4,
            chunk_size = grid_size,  # 整列一个 chunk，无截断传递全部纵向依赖
            headdim    = 64,
        )
        self.ssm_bwd = Mamba2(
            d_model    = d_model,
            d_state    = d_state,
            d_conv     = 4,
            chunk_size = grid_size,
            headdim    = 64,
        )
        # 融合时加非线性，和 RowBiMamba 保持一致扩宽表征
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B, H, W, D)，H=W=174
        B, H, W, D = x.shape

        # 边界防范：确保 ColSSM 中的垂向扫描 chunk_size 能被列高 H 严格整除
        assert H % self.ssm_fwd.chunk_size == 0, f"ColSSM 报错: chunk_size {self.ssm_fwd.chunk_size} 无法整除网格高 H {H}"

        # 转置为列优先：把每列当作独立序列
        x_col = x.permute(0, 2, 1, 3).reshape(B * W, H, D)
        # x_col: (B*174, 174, D)，序列长度=174（行数）

        # 正向从上到下扫描
        fwd = self.ssm_fwd(x_col)                     # (B*174, 174, D)
        # 反向从下到上扫描（需要输入翻转提取关联，最后再次翻转对齐输出坐标映射）
        bwd = self.ssm_bwd(x_col.flip(1)).flip(1)     

        # 特征在深层维度拼接并通过统一融合非线性降维
        out = self.fusion(torch.cat([fwd, bwd], dim=-1))  # (B*174, 174, D)

        # 转置回行优先标准2D网格 (B, H, W, D)
        return out.view(B, W, H, D).permute(0, 2, 1, 3) 


class HierarchicalMambaBlock(nn.Module):
    def __init__(self, d_model=512, d_state_row=64, d_state_col=16, grid_size=174):
        super().__init__()
        # 层次化模块按 "行内双向 -> 列间双向 -> 非线性前馈" 顺序处理
        self.row_mamba = RowBiMamba(d_model, d_state_row, grid_size)
        self.col_ssm   = ColSSM(d_model, d_state_col, grid_size)
        
        # Mamba 的线性 SSM 递归无法完成非线性的处理，此处的 FFN 层负责每位置特征的非线性变换
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # 采用 Pre-Norm 设计，使模型训练更稳定
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1. 行内双向关联（加入残差）
        x = x + self.row_mamba(self.norm1(x))
        # 2. 列间局部关联（加入残差）
        x = x + self.col_ssm(self.norm2(x))
        # 3. 前馈非线性映射（加入残差）
        x = x + self.ffn(self.norm3(x))
        return x


class GeneMamba(nn.Module):
    def __init__(
        self,
        n_genes    = 30000,
        grid_size  = 174,    # 推荐使用 174x174=30276 作为矩阵，解决 173 的不足，剩下的位置填充零向量
        d_model    = 512,
        n_layers   = 8,      # 每一层实际上进行两次 SSM 扫描，8 层相当于普通 mamba 的有效深度 16 层
        d_state_row= 64,
        d_state_col= 16,
        activation_checkpointing=True,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.grid_size = grid_size
        self.activation_checkpointing = bool(activation_checkpointing)

        # 表达值嵌入：标量 → D 维，与基因嵌入信息地位对等
        self.expr_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
        )
        # mask 标志嵌入：标量 → D 维
        self.mask_embed = nn.Linear(1, d_model)

        # 融合投影：三路相加后过投影层
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # 重复堆叠层次化处理 Block
        self.blocks = nn.ModuleList([
            HierarchicalMambaBlock(d_model, d_state_row, d_state_col, grid_size)
            for _ in range(n_layers)
        ])
        
        # 输出头部：为 Flow Matching 用，可以处理被 mask 所在位置的表达值重构等任务（仅作示例结构保留）
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x_feat, x_val, x_mask):
        """
        参数说明:
        由 SG 排列带来的输入信息已天然涵盖了全网格所需，含虚拟补位基因。
        x_feat: (B, 30276, D)
        x_val:  (B, 30276)
        x_mask: (B, 30276)
        """
        B, L, D_feat = x_feat.shape
        H = W = self.grid_size  # 例如 174
        grid_area = H * W       # 174*174 = 30276

        assert L == grid_area, f"输入的序列长 {L} 必须完全等于网格大小 {grid_area} (由上游预处理)"

        # 1. 三路嵌入相加（信息地位对等）
        expr_emb = self.expr_embed(x_val.unsqueeze(-1))    # (B, L, D)
        mask_emb = self.mask_embed(x_mask.unsqueeze(-1))   # (B, L, D)
        x = x_feat + expr_emb + mask_emb                   # (B, L, D)

        # 2. 融合投影
        x = self.input_proj(x)

        # 3. 折叠为二维物理网格：(B, 174, 174, d_model)
        x = x.view(B, H, W, x.shape[-1])

        # 4. 双扫 Mamba 网络提炼全视野空间信息
        # （无论是真实基因特征还是虚拟占位都在此平等通信不加遮蔽）
        for block in self.blocks:
            if torch.is_grad_enabled() and self.activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # 5. 返回还原序列态：(B, 30276, d_model)
        x_out = x.view(B, grid_area, -1)

        return x_out
