import torch
import torch.nn as nn
from .losses import compute_infonce_loss


class HBMLoss(nn.Module):
    """
    Hierarchical Body-part Motion Loss Module
    
    Computes both instance-level and part-level contrastive losses.
    """
    
    def __init__(self, temperature=0.07, lambda_part=1.0, num_parts=6):
        """
        Args:
            temperature: InfoNCE temperature parameter τ
            lambda_part: weight for part-level loss (λ_P in paper)
            num_parts: number of body parts (default: 6)
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_part = lambda_part
        self.num_parts = num_parts
    
    def forward(
        self,
        # Online features
        text_feat,          # (B, D)
        motion_feat_global, # (B, D)
        motion_feat_parts,  # (B, K, D) where K=6
        # Momentum features (positive keys)
        text_feat_m,        # (B, D)
        motion_feat_global_m,  # (B, D)
        motion_feat_parts_m,   # (B, K, D)
        # Queues (negative keys)
        text_queue,         # (D, Q)
        motion_queue,       # (D, Q)
        part_queues,        # (K, D, Q) or list of (D, Q)
    ):
        """
        Compute HBM loss combining instance and part-level alignment.
        
        Returns:
            dict with keys:
                - 'total': total HBM loss
                - 'instance': instance-level loss (L_Inst)
                - 'part': part-level loss (L_Part)
                - 'inst_t2m': text-to-motion instance loss
                - 'inst_m2t': motion-to-text instance loss
                - 'part_t2p': text-to-part loss
                - 'part_p2t': part-to-text loss
        """
        B = text_feat.shape[0]
        
        # ============================================
        # Instance-Level Alignment
        # ============================================
        
        # L_Inst^T2M: text queries, motion keys
        loss_inst_t2m = compute_infonce_loss(
            query=text_feat,                    # (B, D)
            key_positive=motion_feat_global_m,  # (B, D)
            key_negatives=motion_queue,         # (D, Q)
            temperature=self.temperature
        )
        
        # L_Inst^M2T: motion queries, text keys
        loss_inst_m2t = compute_infonce_loss(
            query=motion_feat_global,           # (B, D)
            key_positive=text_feat_m,           # (B, D)
            key_negatives=text_queue,           # (D, Q)
            temperature=self.temperature
        )
        
        loss_instance = loss_inst_t2m + loss_inst_m2t
        
        # ============================================
        # Part-Level Alignment
        # ============================================
        
        # L_Part^T2P: text queries, part keys
        # For each sample i and each part k: InfoNCE(t_i, p̃_{i,k})
        losses_t2p = []
        for k in range(self.num_parts):
            part_feat_m_k = motion_feat_parts_m[:, k, :]  # (B, D)
            # Handle both tensor and list formats for part_queues
            if isinstance(part_queues, list):
                part_queue_k = part_queues[k]  # (D, Q)
            else:
                part_queue_k = part_queues[k]  # (D, Q)
            
            loss_k = compute_infonce_loss(
                query=text_feat,                # (B, D) - same text for all parts
                key_positive=part_feat_m_k,     # (B, D)
                key_negatives=part_queue_k,     # (D, Q)
                temperature=self.temperature
            )
            losses_t2p.append(loss_k)
        
        loss_part_t2p = torch.stack(losses_t2p).mean()  # average over parts
        
        # L_Part^P2T: part queries, text keys
        # For each sample i and each part k: InfoNCE(p_{i,k}, t̃_i)
        losses_p2t = []
        for k in range(self.num_parts):
            part_feat_k = motion_feat_parts[:, k, :]  # (B, D)
            
            loss_k = compute_infonce_loss(
                query=part_feat_k,              # (B, D)
                key_positive=text_feat_m,       # (B, D) - same text for all parts
                key_negatives=text_queue,       # (D, Q)
                temperature=self.temperature
            )
            losses_p2t.append(loss_k)
        
        loss_part_p2t = torch.stack(losses_p2t).mean()  # average over parts
        
        loss_part = loss_part_t2p + loss_part_p2t
        
        # ============================================
        # Total HBM Loss
        # ============================================
        
        loss_total = loss_instance + self.lambda_part * loss_part
        
        return {
            'total': loss_total,
            'instance': loss_instance,
            'part': loss_part,
            'inst_t2m': loss_inst_t2m,
            'inst_m2t': loss_inst_m2t,
            'part_t2p': loss_part_t2p,
            'part_p2t': loss_part_p2t,
        }