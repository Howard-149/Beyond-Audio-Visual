"""Text encoder module for precomputed text embeddings.

Provides temporal modeling for text embeddings (GRU, Transformer, Attention pooling, mean pooling).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    """Temporal encoder for text features.

    Supports multiple architectures:
    - 'gru': Bidirectional GRU with final hidden state
    - 'transformer': Transformer encoder with mean pooling
    - 'attn_pool': Learnable attention-based pooling
    - 'mean_pool': Simple mean pooling over time
    """

    def __init__(
        self,
        input_dim: int,
        model_type: str = "mean_pool",
        gru_hidden: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = True,
        gru_dropout: float = 0.3,
        tr_nhead: int = 4,
        tr_layers: int = 2,
        tr_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        linear_dim: int = 0,
        activation: str = "relu",
        output_classes: int = 0,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.input_dim = input_dim

        if self.model_type == "gru":
            self.backbone = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                batch_first=True,
                dropout=gru_dropout if gru_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.out_dim = gru_hidden * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(gru_dropout)

        elif self.model_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=tr_nhead, dropout=tr_dropout, batch_first=True
            )
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=tr_layers)
            self.out_dim = input_dim
            self.dropout = nn.Dropout(tr_dropout)

        elif self.model_type in ("attn_pool", "weighted_pool", "mean_pool"):
            self.dropout = nn.Dropout(attn_dropout)
            if self.model_type in ("attn_pool", "weighted_pool"):
                self.attn = nn.Linear(input_dim, 1)
            self.out_dim = input_dim

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if linear_dim > 0:
            self.proj = nn.Linear(self.out_dim, linear_dim)
            if activation == "relu":
                self.act = nn.ReLU()
            elif activation == "gelu":
                self.act = nn.GELU()
            else:
                self.act = None
            self.out_dim = linear_dim

        if output_classes > 0:
            self.final_proj = nn.Linear(self.out_dim, output_classes)
            self.out_dim = output_classes

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.model_type == "gru":
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.backbone(packed)
            if self.backbone.bidirectional:
                last_fwd = h_n[-2]
                last_bwd = h_n[-1]
                feat = torch.cat([last_fwd, last_bwd], dim=-1)
            else:
                feat = h_n[-1]
            feat = self.dropout(feat)

        elif self.model_type == "transformer":
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            enc = self.backbone(x, src_key_padding_mask=mask)
            enc = enc.masked_fill(mask.unsqueeze(-1), 0.0)
            sum_enc = enc.sum(dim=1)
            feat = sum_enc / lengths.clamp(min=1).unsqueeze(-1)
            feat = self.dropout(feat)

        elif self.model_type in ("attn_pool", "weighted_pool"):
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            attn_logits = self.attn(x).squeeze(-1)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
            attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
            feat = (attn_weights * x).sum(dim=1)
            feat = self.dropout(feat)

        elif self.model_type == "mean_pool":
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None] 
            x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0) 
            sum_x = x_masked.sum(dim=1)
            feat = sum_x / lengths.clamp(min=1).unsqueeze(-1)
            feat = self.dropout(feat)

        if hasattr(self, "proj"):
            feat = self.proj(feat)
            if hasattr(self, "act") and self.act is not None:
                feat = self.act(feat)

        if hasattr(self, "final_proj"):
            feat = self.final_proj(feat)

        return feat

    def get_output_dim(self) -> int:
        return self.out_dim
