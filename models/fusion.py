"""Fusion modules for combining audio and video representations.

Provides various fusion strategies for multimodal emotion recognition.
"""

import torch
import torch.nn as nn

# Audio feature dimension constants for automatic type detection
MERT_DIM = 768
AF3_DIM = 1280


def align_sequence_to_times(
    seq: torch.Tensor,
    seq_times: torch.Tensor,
    target_times: torch.Tensor,
    device: torch.device,
    window: float = 0.1,
    fill_missing_with_mean: bool = False,
) -> torch.Tensor:
    """Align sequence to target timestamps in seconds (real time). Vectorized.

    For each target time, pool embeddings within [t-window, t+window]. When the window
    is empty: if fill_missing_with_mean=True (e.g. video with missing frames), use
    mean(seq) over all positions; else use nearest by time.
    """
    seq_len = seq.size(0)
    if seq_len <= 0:
        return seq
    T = target_times.size(0)
    window_mask = (
        (seq_times.unsqueeze(0) >= target_times.unsqueeze(1) - window)
        & (seq_times.unsqueeze(0) <= target_times.unsqueeze(1) + window)
    )
    w = window_mask.float()
    sum_w = w.sum(dim=1)
    pooled_window = (w @ seq) / sum_w.clamp(min=1e-9).unsqueeze(1)
    if fill_missing_with_mean:
        seq_mean = seq.mean(dim=0, keepdim=True).expand(T, -1)
    else:
        diff = (seq_times.unsqueeze(0) - target_times.unsqueeze(1)).abs()
        nearest_idx = diff.argmin(dim=1)
        seq_mean = seq[nearest_idx]
    aligned = torch.where(sum_w.unsqueeze(1) > 0, pooled_window, seq_mean)
    return aligned


class ConcatFusion(nn.Module):
    """Simple concatenation fusion.

    Concatenates available modality features (audio, video, lyrics, comments).
    """

    def __init__(
        self,
        audio_dim: int = 0,
        video_dim: int = 0,
        lyrics_dim: int = 0,
        comments_dim: int = 0,
        num_classes: int = 5,
        fusion_config: dict | None = None,
    ):
        super().__init__()
        cfg = fusion_config or {}
        hidden_dim = cfg.get("hidden_dim", 0)
        dropout = cfg.get("dropout", 0.5)
        self.modality_dims = {}
        if audio_dim > 0:
            self.modality_dims["audio"] = audio_dim
        if video_dim > 0:
            self.modality_dims["video"] = video_dim
        if lyrics_dim > 0:
            self.modality_dims["lyrics"] = lyrics_dim
        if comments_dim > 0:
            self.modality_dims["comments"] = comments_dim

        if not self.modality_dims:
            raise ValueError("At least one modality dimension must be > 0")

        self.concat_dim = sum(self.modality_dims.values())
        self.dropout = nn.Dropout(dropout)

        if hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.concat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.fc = nn.Linear(self.concat_dim, num_classes)

    def forward(self, modality_feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """modality_feats: dict with keys in {'audio', 'video', 'lyrics', 'comments'}, values (B, D)."""
        provided = set(modality_feats.keys())
        expected = set(self.modality_dims.keys())
        if provided != expected:
            raise ValueError(f"Modality mismatch: expected {expected}, got {provided}")

        feats = []
        for mod in ["audio", "video", "lyrics", "comments"]:
            if mod in modality_feats:
                feats.append(modality_feats[mod])
        fused = torch.cat(feats, dim=-1)
        fused = self.dropout(fused)
        logits = self.fc(fused)
        return logits


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for sequence features.

    Supports aligned audio-video cross-attention with optional bidirectionality.
    Q/K/V modality selection is configurable for the primary attention path.
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        lyrics_dim: int = 0,
        comments_dim: int = 0,
        num_classes: int = 5,
        mode: str = "cross_attention",
        q_modality: str = "video",
        k_modality: str = "audio",
        v_modality: str = "audio",
        bidirectional: bool = True,
        video_fps: float = 30.0,
        mert_seconds_per_emb: float = 0.0133,
        af3_seconds_per_emb: float = 0.04,
        fusion_config: dict | None = None,
    ):
        super().__init__()
        cfg = fusion_config or {}
        cross_attn_heads = cfg.get("cross_attn_heads", 4)
        dropout = cfg.get("dropout", 0.1)
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.lyrics_dim = lyrics_dim
        self.comments_dim = comments_dim
        self.num_classes = num_classes
        self.mode = mode
        self.q_modality = q_modality
        self.k_modality = k_modality
        self.v_modality = v_modality
        self.bidirectional = bidirectional
        self.video_fps = video_fps
        self.mert_seconds_per_emb = mert_seconds_per_emb
        self.af3_seconds_per_emb = af3_seconds_per_emb

        # Primary attention (q, k, v)
        q_dim = self._get_dim(q_modality)
        k_dim = self._get_dim(k_modality)
        v_dim = self._get_dim(v_modality)
        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            kdim=k_dim,
            vdim=v_dim,
            num_heads=cross_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Reverse attention for bidirectional (only if Q and K/V are different modalities)
        self.reverse_attn = None
        if self.bidirectional and q_modality != k_modality:
            self.reverse_attn = nn.MultiheadAttention(
                embed_dim=k_dim,
                kdim=q_dim,
                vdim=q_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True,
            )

        out_dim = q_dim + (k_dim if self.reverse_attn is not None else 0)
        self.classifier = nn.Linear(out_dim, num_classes)

    def _get_dim(self, modality: str) -> int:
        if modality == "audio":
            return self.audio_dim
        if modality == "video":
            return self.video_dim
        if modality == "lyrics":
            if self.lyrics_dim <= 0:
                raise ValueError("lyrics_dim must be > 0 when using lyrics modality")
            return self.lyrics_dim
        if modality == "comments":
            if self.comments_dim <= 0:
                raise ValueError("comments_dim must be > 0 when using comments modality")
            return self.comments_dim
        raise ValueError(f"Unknown modality: {modality}")

    def forward(
        self,
        modality_seqs: dict[str, torch.Tensor],
        modality_lengths: dict[str, torch.Tensor],
        video_indices: list | None = None,
    ) -> torch.Tensor:
        """Forward with flexible modality sequences. Aligns by real time (seconds).
        Timeline max_end is from video/audio only; text has no real time and is aligned
        passively as uniform over [0, max_end]. Video missing frames filled with mean(seq)."""
        if self.mode not in ("cross_attention", "cross_attention_mean"):
            raise ValueError(f"Unsupported mode: {self.mode}")

        required_mods = {self.q_modality, self.k_modality, self.v_modality}
        provided_mods = set(modality_seqs.keys())
        if not required_mods.issubset(provided_mods):
            raise ValueError(
                f"Missing modalities: required {required_mods}, got {provided_mods}"
            )

        q_x = modality_seqs[self.q_modality]
        k_x = modality_seqs[self.k_modality]
        v_x = modality_seqs[self.v_modality]
        q_lengths = modality_lengths[self.q_modality]
        k_lengths = modality_lengths[self.k_modality]
        v_lengths = modality_lengths[self.v_modality]

        B = q_x.size(0)
        device = q_x.device
        out_feats = []

        for i in range(B):
            q_len = int(q_lengths[i].item())
            k_len = int(k_lengths[i].item())
            v_len = int(v_lengths[i].item())
            q_seq = q_x[i, :q_len]
            k_seq = k_x[i, :k_len]
            v_seq = v_x[i, :v_len]

            q_times = self._seq_times_for_modality(
                self.q_modality, q_len, i, video_indices, device, q_x
            )
            k_times = self._seq_times_for_modality(
                self.k_modality, k_len, i, video_indices, device, k_x
            )
            if self.k_modality == self.v_modality:
                v_times = k_times
            else:
                v_times = self._seq_times_for_modality(
                    self.v_modality, v_len, i, video_indices, device, v_x
                )

            # Timeline from video/audio only; lyrics/comments passive (uniform over [0, max_end]).
            text_like = {"lyrics", "comments"}
            valid_times = [t for mod, t in [
                (self.q_modality, q_times), (self.k_modality, k_times), (self.v_modality, v_times)
            ] if mod not in text_like and len(t) > 0]
            if not valid_times:
                continue
            max_end = max(t[-1].item() for t in valid_times)
            max_steps = max(q_len, k_len, v_len)
            if max_steps <= 0:
                continue
            target_times = torch.linspace(0.0, max_end, max_steps, device=device)

            if self.q_modality in text_like and q_len > 0:
                q_times = torch.linspace(0.0, max_end, q_len, device=device)
            if self.k_modality in text_like and k_len > 0:
                k_times = torch.linspace(0.0, max_end, k_len, device=device)
            if self.v_modality in text_like and v_len > 0:
                v_times = torch.linspace(0.0, max_end, v_len, device=device)

            q_aligned = align_sequence_to_times(
                q_seq, q_times, target_times, device, window=0.1,
                fill_missing_with_mean=(self.q_modality == "video"),
            )
            k_aligned = align_sequence_to_times(
                k_seq, k_times, target_times, device, window=0.1,
                fill_missing_with_mean=(self.k_modality == "video"),
            )
            v_aligned = k_aligned if self.k_modality == self.v_modality else align_sequence_to_times(
                v_seq, v_times, target_times, device, window=0.1,
                fill_missing_with_mean=(self.v_modality == "video"),
            )

            attn_out, _ = self.attn(
                query=q_aligned.unsqueeze(0),
                key=k_aligned.unsqueeze(0),
                value=v_aligned.unsqueeze(0),
            )
            attn_feat = attn_out.mean(dim=1)

            if self.reverse_attn is not None:
                attn_out_rev, _ = self.reverse_attn(
                    query=k_aligned.unsqueeze(0),
                    key=q_aligned.unsqueeze(0),
                    value=q_aligned.unsqueeze(0),
                )
                attn_feat_rev = attn_out_rev.mean(dim=1)
                attn_feat = torch.cat([attn_feat, attn_feat_rev], dim=-1)

            out_feats.append(attn_feat.squeeze(0))

        out_feats = torch.stack(out_feats, dim=0)
        logits = self.classifier(out_feats)
        return logits

    def _seq_times_for_modality(
        self,
        modality: str,
        seq_len: int,
        batch_idx: int,
        video_indices: list | None,
        device: torch.device,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return (seq_len,) timestamps in seconds. Video uses frame indices. Lyrics/comments return placeholder (caller overwrites)."""
        if modality == "video":
            if video_indices is not None and batch_idx < len(video_indices) and video_indices[batch_idx] is not None and len(video_indices[batch_idx]) > 0:
                idx = video_indices[batch_idx][:seq_len]
                return torch.tensor(idx, dtype=torch.float32, device=device) / self.video_fps
            return torch.arange(seq_len, dtype=torch.float32, device=device) / self.video_fps
        if modality == "audio":
            sec = self.af3_seconds_per_emb if x.shape[-1] == AF3_DIM else self.mert_seconds_per_emb
            return torch.arange(seq_len, dtype=torch.float32, device=device) * sec
        if modality in ("lyrics", "comments"):
            return torch.linspace(0.0, 1.0, seq_len, device=device)
        raise ValueError(f"Unknown modality: {modality}")



class VideoQueryAttentionFusion(nn.Module):
    """Video-query cross-attention fusion aligning all modalities to shared timestamps.
    Lyrics and comments are two separate text modalities (either or both can be 0)."""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        lyrics_dim: int = 0,
        comments_dim: int = 0,
        num_classes: int = 5,
        video_fps: float = 30.0,
        window: float = 0.1,
        fusion_config: dict | None = None,
    ):
        super().__init__()
        cfg = fusion_config or {}
        hidden_dim = cfg.get("hidden_dim", 0)
        attn_heads = cfg.get("cross_attn_heads", 4)
        dropout = cfg.get("dropout", 0.1)

        if video_dim <= 0 or audio_dim <= 0:
            raise ValueError("Video-query cross attention requires video_dim and audio_dim > 0")
        if lyrics_dim <= 0 and comments_dim <= 0:
            raise ValueError("At least one of lyrics_dim or comments_dim must be > 0")

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.lyrics_dim = lyrics_dim
        self.comments_dim = comments_dim
        self.video_fps = video_fps
        self.window = window

        self.audio_attn = nn.MultiheadAttention(
            embed_dim=video_dim,
            kdim=audio_dim,
            vdim=audio_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.lyrics_attn = None
        if lyrics_dim > 0:
            self.lyrics_attn = nn.MultiheadAttention(
                embed_dim=video_dim,
                kdim=lyrics_dim,
                vdim=lyrics_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.comments_attn = None
        if comments_dim > 0:
            self.comments_attn = nn.MultiheadAttention(
                embed_dim=video_dim,
                kdim=comments_dim,
                vdim=comments_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)
        num_ctx = 2 + (1 if lyrics_dim > 0 else 0) + (1 if comments_dim > 0 else 0)
        network_in = video_dim * num_ctx
        if hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(network_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.fc = nn.Linear(network_in, num_classes)

    def forward(
        self,
        video_seq: torch.Tensor,
        video_len: torch.Tensor,
        audio_seq: torch.Tensor | None,
        audio_len: torch.Tensor | None,
        lyrics_seq: torch.Tensor | None,
        lyrics_len: torch.Tensor | None,
        comments_seq: torch.Tensor | None,
        comments_len: torch.Tensor | None,
        video_indices: list | None = None,
    ) -> torch.Tensor:
        """Align each modality, run video-query cross attention, concatenate contexts."""

        B = video_seq.size(0)
        device = video_seq.device
        audio_sec = self._get_audio_seconds_per_emb()

        q_list, audio_list, lyrics_list, comments_list, valid_lens = [], [], [], [], []
        for i in range(B):
            v_len = int(video_len[i].item())
            if v_len == 0:
                raise RuntimeError("Video sequences must have positive length for video_query_cross_attention")
            a_len = int(audio_len[i].item()) if audio_len is not None else 0
            l_len = int(lyrics_len[i].item()) if lyrics_len is not None else 0
            c_len = int(comments_len[i].item()) if comments_len is not None else 0

            v_seq = video_seq[i, :v_len]
            a_seq = audio_seq[i, :a_len] if audio_seq is not None and a_len > 0 else None
            l_seq = lyrics_seq[i, :l_len] if lyrics_seq is not None and l_len > 0 else None
            c_seq = comments_seq[i, :c_len] if comments_seq is not None and c_len > 0 else None

            target_times = self._build_target_times(
                v_len, a_len, l_len, c_len,
                video_indices[i] if video_indices is not None else None,
                device,
            )
            T = target_times.size(0)
            valid_lens.append(T)

            if video_indices is not None and video_indices[i] is not None and len(video_indices[i]) > 0:
                v_times = torch.tensor(video_indices[i][:v_len], dtype=torch.float32, device=device) / self.video_fps
            else:
                v_times = torch.arange(v_len, dtype=torch.float32, device=device) / self.video_fps
            q_aligned = align_sequence_to_times(
                v_seq, v_times, target_times, device, window=self.window,
                fill_missing_with_mean=True,
            )
            q_list.append(q_aligned)

            if a_seq is not None:
                a_times = torch.arange(a_len, dtype=torch.float32, device=device) * audio_sec
                audio_aligned = align_sequence_to_times(a_seq, a_times, target_times, device, window=self.window)
                audio_list.append(audio_aligned)
            else:
                audio_list.append(torch.zeros(T, self.audio_dim, device=device))

            max_end = target_times[-1].item()
            if l_seq is not None and self.lyrics_dim > 0:
                l_times = torch.linspace(0.0, max_end, l_len, device=device)
                lyrics_aligned = align_sequence_to_times(l_seq, l_times, target_times, device, window=self.window)
                lyrics_list.append(lyrics_aligned)
            elif self.lyrics_attn is not None:
                lyrics_list.append(torch.zeros(T, self.lyrics_dim, device=device))

            if c_seq is not None and self.comments_dim > 0:
                c_times = torch.linspace(0.0, max_end, c_len, device=device)
                comments_aligned = align_sequence_to_times(c_seq, c_times, target_times, device, window=self.window)
                comments_list.append(comments_aligned)
            elif self.comments_attn is not None:
                comments_list.append(torch.zeros(T, self.comments_dim, device=device))

        max_T = max(valid_lens)
        q_padded = torch.nn.utils.rnn.pad_sequence(q_list, batch_first=True, padding_value=0.0)
        audio_padded = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        valid_lens_t = torch.tensor(valid_lens, device=device, dtype=torch.long)
        valid_mask = torch.arange(max_T, device=device).unsqueeze(0) < valid_lens_t.unsqueeze(1)
        key_padding_mask = ~valid_mask

        attn_audio, _ = self.audio_attn(
            q_padded, audio_padded, audio_padded, key_padding_mask=key_padding_mask
        ) 
        v_mask = valid_mask.unsqueeze(-1).float()
        video_context = (q_padded * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1e-9)
        audio_context = (attn_audio * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1e-9)
        context_list = [video_context, audio_context]

        if self.lyrics_attn is not None and lyrics_list:
            lyrics_padded = torch.nn.utils.rnn.pad_sequence(lyrics_list, batch_first=True, padding_value=0.0)
            attn_lyrics, _ = self.lyrics_attn(
                q_padded, lyrics_padded, lyrics_padded, key_padding_mask=key_padding_mask
            )
            lyrics_context = (attn_lyrics * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1e-9) # mean pooling
            context_list.append(lyrics_context)
        if self.comments_attn is not None and comments_list:
            comments_padded = torch.nn.utils.rnn.pad_sequence(comments_list, batch_first=True, padding_value=0.0)
            attn_comments, _ = self.comments_attn(
                q_padded, comments_padded, comments_padded, key_padding_mask=key_padding_mask
            )
            comments_context = (attn_comments * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1e-9)
            context_list.append(comments_context)

        merged = torch.cat(context_list, dim=-1)
        merged = self.dropout(merged)
        logits = self.fc(merged)
        return logits

    def _build_target_times(
        self,
        v_len: int,
        a_len: int,
        l_len: int,
        c_len: int,
        indices: list | None,
        device: torch.device,
    ) -> torch.Tensor:
        times = []
        if indices is not None and len(indices) > 0:
            idx = indices[:v_len]
            times.append(torch.tensor(idx, dtype=torch.float32, device=device) / self.video_fps)
        else:
            times.append(torch.arange(v_len, dtype=torch.float32, device=device) / self.video_fps)
        if a_len > 0:
            audio_seconds = self._get_audio_seconds_per_emb()
            times.append(torch.arange(a_len, dtype=torch.float32, device=device) * audio_seconds)
        max_end = max([t[-1].item() for t in times]) if times else 0.0
        max_steps = max(v_len, a_len, l_len, c_len)
        return torch.linspace(0, max_end, max_steps, device=device)

    def _get_audio_seconds_per_emb(self) -> float:
        feat_dim = self.audio_dim
        if feat_dim == MERT_DIM:
            return 0.0133
        if feat_dim == AF3_DIM:
            return 0.04
        return 0.0133


