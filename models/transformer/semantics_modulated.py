import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# @ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Normalization
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        
        # Query projection (只作用于motion tokens z)
        self.query = nn.Linear(latent_dim, latent_dim)
        
        # Info fusion MLP: [t; R_m; R_t] -> info
        # t, R_m, R_t 都是 latent_dim 维度
        self.info_mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Key projection (作用于 [z, info])
        self.key = nn.Linear(latent_dim, latent_dim)
        
        # Value projection (作用于 [z, R_m], 只包含motion-domain)
        self.value = nn.Linear(latent_dim, latent_dim)
    
        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf, src_mask, cond_type, re_dict=None):
        """
        参数:
            x: (B, N, D) - motion tokens z
            xf: (B, 1, D) - text prompt t
            src_mask: (B, N, 1) - valid position mask
            cond_type: (B, 1, 1) - 控制是否使用条件
            re_dict: {'re_motion': (B, K, 1, D), 're_text': (B, K, 1, D)}
        """

        B, N, D = x.shape
        H = self.num_heads

        # 提取retrieval特征
        re_motion = re_dict['re_motion'].squeeze(2)  # (B, K, D)
        re_text = re_dict['re_text'].squeeze(2)      # (B, K, D)

        # 条件控制
        text_cond = (cond_type % 10 > 0).float()     # 是否使用text
        retr_cond = (cond_type // 10 > 0).float()    # 是否使用retrieval

        # Query
        # Q = W_q · z
        z_norm = self.norm(x)
        Q = self.query(z_norm) * src_mask  # (B, N, D)

        # ========== Info Fusion ==========
        # info = MLP([t; R_m; R_t])
        # 将K个retrieval取平均或选第一个
        R_m_pooled = re_motion.mean(dim=1, keepdim=True)  # (B, 1, D)
        R_t_pooled = re_text.mean(dim=1, keepdim=True)    # (B, 1, D)

        # 拼接 [t; R_m; R_t]
        fused_input = torch.cat([
            xf * text_cond,           # (B, 1, D) 
            R_m_pooled * retr_cond,   # (B, 1, D)
            R_t_pooled * retr_cond    # (B, 1, D)
        ], dim=-1)  # (B, 1, 3D)
        
        info = self.info_mlp(fused_input)  # (B, 1, D)

        # ========== Key ==========
        # K = W_k · concat([z, info])
        kv_input = torch.cat([z_norm, info], dim=1)  # (B, N+1, D)
        K = self.key(kv_input)  # (B, N+1, D)

        # ========== Value ==========
        # V = W_v · concat([z, R_m])
        # 这里只使用motion-domain特征
        V_input = torch.cat([z_norm, R_m_pooled], dim=1)  # (B, N+1, D)
        V = self.value(V_input)  # (B, N+1, D)

        # ========== Multi-Head Attention ==========
        # Reshape: (B, seq, D) -> (B, H, seq, head_dim)
        Q = Q.view(B, N, H, self.head_dim).transpose(1, 2)       # (B, H, N, head_dim)
        K = K.view(B, N+1, H, self.head_dim).transpose(1, 2)     # (B, H, N+1, head_dim)
        V = V.view(B, N+1, H, self.head_dim).transpose(1, 2)     # (B, H, N+1, head_dim)

        # Scaled Dot-Product: QK^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N+1)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N+1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # (B, H, N, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N, H, head_dim)
        attn_output = attn_output.view(B, N, D)  # (B, N, D)
        
        # Output projection
        output = self.out_proj(attn_output)

        # 残差连接（在外层trans_forward中处理）
        return output