"""Video encoder module for precomputed video/face embeddings.

Provides LSTM-based sequence modeling. Logic originally lived in the legacy VER
unimodal trainer; video-only runs now use
`multimodal_classifier.py --fusion-type video_only`.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VideoEncoder(nn.Module):
    """LSTM-based encoder for video/face features.
    
    Can handle both single expert (T, D) and multi-expert (T, E, D) features.
    """
    
    def __init__(
        self,
        feat_dim: int,
        lstm_hidden: int = 256,
        dropout: float = 0.5,
        num_experts: int = 1,
        final_lstm_hidden: int = 256,
        output_classes: int = 0,
    ):
        """Initialize video encoder.
        
        Args:
            feat_dim: dimension of input video features per expert
            lstm_hidden: hidden size for LSTM
            dropout: dropout rate
            num_experts: number of expert features (1 for single, >1 for multi-expert)
            final_lstm_hidden: hidden size for final fusion LSTM (multi-expert only)
            output_classes: number of output classes (0 = no final classification layer)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.lstm_hidden = lstm_hidden
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout)
        
        if num_experts == 1:
            # Simple single-expert LSTM
            self.lstm = nn.LSTM(feat_dim, lstm_hidden, num_layers=1, batch_first=True)
            self.out_dim = lstm_hidden
        else:
            # Multi-expert: independent LSTM per expert + fusion LSTM
            self.expert_lstms = nn.ModuleList([
                nn.LSTM(feat_dim, lstm_hidden, batch_first=True)
                for _ in range(num_experts)
            ])
            self.final_lstm = nn.LSTM(
                num_experts * lstm_hidden, final_lstm_hidden, batch_first=True
            )
            self.final_lstm_hidden = final_lstm_hidden
            self.out_dim = final_lstm_hidden
        
        # Optional output projection
        if output_classes > 0:
            self.final_proj = nn.Linear(self.out_dim, output_classes)
            self.out_dim = output_classes
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, T, D) for single expert or (B, T, E, D) for multi-expert
            lengths: (B,) valid lengths for each sequence
        
        Returns:
            (B, out_dim) encoded video features
        """
        if self.num_experts == 1:
            # Simple single-expert path
            if x.dim() == 4:
                # (B, T, 1, D) -> (B, T, D)
                x = x.squeeze(2)
            
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            feat = h_n[-1]  # (B, lstm_hidden)
            feat = self.dropout(feat)
        else:
            # Multi-expert path
            if x.dim() == 3:
                # (B, T, D) -> (B, T, 1, D)
                x = x.unsqueeze(2)
            
            B, T, E, D = x.size()
            
            # Process each expert separately
            outs = []
            for ei in range(E):
                xi = x[:, :, ei, :]  # (B, T, D)
                packed = pack_padded_sequence(
                    xi, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.expert_lstms[ei](packed)
                out_padded_e, _ = pad_packed_sequence(
                    packed_out, batch_first=True, total_length=T
                )
                outs.append(out_padded_e)  # (B, T, lstm_hidden)
            
            # Concatenate expert outputs
            concat = torch.cat(outs, dim=-1)  # (B, T, E * lstm_hidden)
            
            # Final fusion LSTM
            packed_final = pack_padded_sequence(
                concat, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n_f, _) = self.final_lstm(packed_final)
            feat = h_n_f[-1]  # (B, final_lstm_hidden)
            feat = self.dropout(feat)
        
        # Optional final projection
        if hasattr(self, "final_proj"):
            feat = self.final_proj(feat)
        
        return feat
    
    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.out_dim
