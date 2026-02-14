import torch
import torch.nn.functional as F


def compute_infonce_loss(query, key_positive, key_negatives, temperature=0.07):
    """
    Compute InfoNCE loss for contrastive learning.
    
    Args:
        query: (B, D) - query features
        key_positive: (B, D) - positive key features (from momentum encoder)
        key_negatives: (D, K) - negative keys from queue
        temperature: temperature parameter τ
    
    Returns:
        loss: scalar InfoNCE loss
    
    Mathematical formulation:
        For each query q_i:
            positive_sim = sim(q_i, k̃_i) / τ
            negative_sims = sim(q_i, k⁻) / τ  for all k⁻ in queue
            
            InfoNCE = -log[exp(positive_sim) / (exp(positive_sim) + Σ exp(negative_sims))]
                    = -positive_sim + log_sum_exp([positive_sim, negative_sims])
    """
    B, D = query.shape
    K = key_negatives.shape[1]
    
    # Normalize features
    query = F.normalize(query, dim=-1)
    key_positive = F.normalize(key_positive, dim=-1)
    key_negatives = F.normalize(key_negatives, dim=0)
    
    # Compute positive similarities: (B,)
    positive_sim = torch.sum(query * key_positive, dim=1) / temperature  # (B,)
    
    # Compute negative similarities: (B, K)
    negative_sims = query @ key_negatives / temperature  # (B, D) @ (D, K) = (B, K)
    
    # Concatenate positive and negative logits: (B, 1+K)
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sims], dim=1)  # (B, 1+K)
    
    # InfoNCE loss: -log[exp(positive) / sum(exp(all))]
    # = -positive + log_sum_exp(all)
    # = cross_entropy with label=0
    labels = torch.zeros(B, dtype=torch.long, device=query.device)  # positive is at index 0
    loss = F.cross_entropy(logits, labels)
    
    return loss


def compute_infonce_loss_with_queue(query, key_positive, queue, temperature=0.07):
    """
    Convenience wrapper that handles queue directly.
    
    Args:
        query: (B, D)
        key_positive: (B, D)
        queue: (D, K)
        temperature: float
    
    Returns:
        loss: scalar
    """
    return compute_infonce_loss(query, key_positive, queue, temperature)