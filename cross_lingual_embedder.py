"""
Cross-Lingual Multimodal Shared Embedding Alignment Engine:
E_Santali <-> E_Bengali <-> E_English with InfoNCE Contrastive Loss and Procrustes Alignment.
"""

import math
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CrossLingualProjectionHead(nn.Module):
        """
        Multilingual Contrastive Projection Network mapping language-specific embeddings
        into a unified, normalized hypersphere for semantic alignment.
        """
        def __init__(self, input_dim: int = 1024, shared_dim: int = 768, temperature: float = 0.07):
            super().__init__()
            self.input_dim = input_dim
            self.shared_dim = shared_dim
            self.temperature = nn.Parameter(torch.tensor(temperature))

            # Non-linear projection MLP with LayerNorm
            self.projection = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Linear(shared_dim, shared_dim),
            )

        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """Project and L2-normalize embeddings."""
            projected = self.projection(embeddings)
            return F.normalize(projected, p=2, dim=-1)

        def compute_infonce_loss(
            self,
            source_embeds: torch.Tensor,
            target_embeds: torch.Tensor,
        ) -> torch.Tensor:
            """
            Bidirectional InfoNCE contrastive alignment loss:
            L = 0.5 * (L(src -> tgt) + L(tgt -> src))
            """
            z_src = self.forward(source_embeds)
            z_tgt = self.forward(target_embeds)

            # Cosine similarity matrix scaled by temperature
            logits = torch.matmul(z_src, z_tgt.T) / torch.clamp(self.temperature, min=1e-4)

            batch_size = source_embeds.size(0)
            labels = torch.arange(batch_size, device=source_embeds.device)

            loss_src_to_tgt = F.cross_entropy(logits, labels)
            loss_tgt_to_src = F.cross_entropy(logits.T, labels)

            return 0.5 * (loss_src_to_tgt + loss_tgt_to_src)


    class CrossAttentionCrossLingualBridge(nn.Module):
        """
        Cross-Attention Transformer layer projecting source language (e.g. Ol Chiki Santali)
        context representations into target language (Bengali / English) latent space.
        """
        def __init__(self, d_model: int = 768, nhead: int = 12, dropout: float = 0.1):
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(
            self,
            target_queries: torch.Tensor,
            source_keys_values: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            attn_output, _ = self.multihead_attn(
                query=target_queries,
                key=source_keys_values,
                value=source_keys_values,
                key_padding_mask=key_padding_mask,
            )
            x = self.layer_norm(target_queries + attn_output)
            out = self.norm2(x + self.ffn(x))
            return out


class ProcrustesAligner:
    """
    Closed-form Orthogonal Procrustes Transformation for cross-lingual vector spaces:
    Finds W* in O(d) minimizing || W * X_source - X_target ||_F.
    Solution: SVD(X_target @ X_source.T) = U @ Sigma @ V.T -> W* = U @ V.T.
    """

    @staticmethod
    def align_numpy(source_matrix, target_matrix):
        """Pure NumPy closed-form orthogonal Procrustes computation."""
        import numpy as np
        # M = Target @ Source.T
        m = np.matmul(target_matrix.T, source_matrix)
        u, _, vt = np.linalg.svd(m)
        w_star = np.matmul(u, vt)
        return w_star
