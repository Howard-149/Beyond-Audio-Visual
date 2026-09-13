#!/usr/bin/env python3
"""Train / eval multimodal emotion recognition on EmoMV feature manifests.

Expects modality roots with ``{train,val,test}_features.csv`` (see README).
CLI examples and the song-disjoint split workflow live in the repo README.
"""

MERT_AUDIO_SECONDS_PER_EMBEDDING = 0.0133
AF3_AUDIO_SECONDS_PER_EMBEDDING = 0.04
VIDEO_FPS = 30
MERT_DIM = 768
AF3_DIM = 1280

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.multimodal import EmoMVMultimodalDataset, collate_multimodal
from VER.datasets.datasets import EMOMV_CLASS_LABELS
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from models.fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    VideoQueryAttentionFusion,
)


class MultimodalEmotionClassifier(nn.Module):
    """End-to-end multimodal emotion classifier with late concat / cross-attn fusion."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        lyrics_dim: int,
        comments_dim: int,
        audio_config: dict,
        video_config: dict,
        lyrics_config: dict,
        comments_config: dict,
        fusion_type: str = "late_concat",
        fusion_config: Optional[dict] = None,
        num_classes: int = 5,
        cross_attn_q: str = "video",
        cross_attn_k: str = "audio",
        cross_attn_v: str = "audio",
        cross_attn_bidirectional: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        fusion_config = fusion_config or {}

        if fusion_type == "late_concat":
            audio_out_dim = 0
            if audio_dim > 0:
                self.audio_encoder = AudioEncoder(input_dim=audio_dim, **audio_config)
                if hasattr(self.audio_encoder, "get_output_dim"):
                    audio_out_dim = self.audio_encoder.get_output_dim()
                else:
                    raise RuntimeError("AudioEncoder must implement get_output_dim() for late fusion.")
            video_out_dim = 0
            if video_dim > 0:
                self.video_encoder = VideoEncoder(feat_dim=video_dim, **video_config)
                if hasattr(self.video_encoder, "get_output_dim"):
                    video_out_dim = self.video_encoder.get_output_dim()
                else:
                    raise RuntimeError("VideoEncoder must implement get_output_dim() for late fusion.")
            lyrics_out_dim = 0
            if lyrics_dim > 0:
                self.lyrics_encoder = TextEncoder(input_dim=lyrics_dim, **lyrics_config)
                if hasattr(self.lyrics_encoder, "get_output_dim"):
                    lyrics_out_dim = self.lyrics_encoder.get_output_dim()
                else:
                    raise RuntimeError("TextEncoder must implement get_output_dim() for late fusion.")
            comments_out_dim = 0
            if comments_dim > 0:
                self.comments_encoder = TextEncoder(input_dim=comments_dim, **comments_config)
                if hasattr(self.comments_encoder, "get_output_dim"):
                    comments_out_dim = self.comments_encoder.get_output_dim()
                else:
                    raise RuntimeError("TextEncoder must implement get_output_dim() for late fusion.")

            self.fusion = ConcatFusion(
                audio_dim=audio_out_dim,
                video_dim=video_out_dim,
                lyrics_dim=lyrics_out_dim,
                comments_dim=comments_out_dim,
                num_classes=num_classes,
                fusion_config=fusion_config,
            )
        elif fusion_type == "audio_only":
            self.audio_encoder = AudioEncoder(input_dim=audio_dim, **audio_config)
            if hasattr(self.audio_encoder, "get_output_dim"):
                audio_out_dim = self.audio_encoder.get_output_dim()
            else:
                raise RuntimeError("AudioEncoder must implement get_output_dim() for audio_only.")
            self.classifier = nn.Linear(audio_out_dim, num_classes)
        elif fusion_type == "video_only":
            self.video_encoder = VideoEncoder(feat_dim=video_dim, **video_config)
            if hasattr(self.video_encoder, "get_output_dim"):
                video_out_dim = self.video_encoder.get_output_dim()
            else:
                raise RuntimeError("VideoEncoder must implement get_output_dim() for video_only.")
            self.classifier = nn.Linear(video_out_dim, num_classes)
        elif fusion_type == "lyrics_only":
            self.lyrics_encoder = TextEncoder(input_dim=lyrics_dim, **lyrics_config)
            lyrics_out_dim = self.lyrics_encoder.get_output_dim()
            self.classifier = nn.Linear(lyrics_out_dim, num_classes)
        elif fusion_type == "comments_only":
            self.comments_encoder = TextEncoder(input_dim=comments_dim, **comments_config)
            comments_out_dim = self.comments_encoder.get_output_dim()
            self.classifier = nn.Linear(comments_out_dim, num_classes)
        elif fusion_type == "video_query_cross_attention":
            if audio_dim <= 0 or video_dim <= 0:
                raise RuntimeError("video_query_cross_attention requires audio and video dimensions > 0")
            if lyrics_dim <= 0 and comments_dim <= 0:
                raise RuntimeError("video_query_cross_attention requires at least one of lyrics_dim or comments_dim > 0")
            self.fusion = VideoQueryAttentionFusion(
                video_dim=video_dim,
                audio_dim=audio_dim,
                lyrics_dim=lyrics_dim,
                comments_dim=comments_dim,
                num_classes=num_classes,
                video_fps=30.0,
                fusion_config=fusion_config,
            )
        elif fusion_type in ("cross_attention", "cross_attention_mean"):
            self.fusion = CrossAttentionFusion(
                audio_dim=audio_dim,
                video_dim=video_dim,
                lyrics_dim=lyrics_dim,
                comments_dim=comments_dim,
                num_classes=num_classes,
                mode=fusion_type,
                q_modality=cross_attn_q,
                k_modality=cross_attn_k,
                v_modality=cross_attn_v,
                bidirectional=cross_attn_bidirectional,
                fusion_config=fusion_config,
            )

    def forward(
        self,
        video_x: torch.Tensor,
        audio_x: torch.Tensor,
        lyrics_x: torch.Tensor | None,
        comments_x: torch.Tensor | None,
        video_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        lyrics_lengths: torch.Tensor | None,
        comments_lengths: torch.Tensor | None,
        video_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Forward pass. Lyrics and comments are separate modalities."""
        if self.fusion_type == "audio_only":
            audio_feat = self.audio_encoder(audio_x, audio_lengths)
            logits = self.classifier(audio_feat)
            return logits
        elif self.fusion_type == "video_only":
            video_feat = self.video_encoder(video_x, video_lengths)
            logits = self.classifier(video_feat)
            return logits
        elif self.fusion_type == "lyrics_only":
            if lyrics_x is None or lyrics_lengths is None:
                raise RuntimeError("lyrics_x and lyrics_lengths required for lyrics_only")
            lyrics_feat = self.lyrics_encoder(lyrics_x, lyrics_lengths)
            logits = self.classifier(lyrics_feat)
            return logits
        elif self.fusion_type == "comments_only":
            if comments_x is None or comments_lengths is None:
                raise RuntimeError("comments_x and comments_lengths required for comments_only")
            comments_feat = self.comments_encoder(comments_x, comments_lengths)
            logits = self.classifier(comments_feat)
            return logits
        elif self.fusion_type == "late_concat":
            modality_feats = {}
            if hasattr(self, "audio_encoder") and audio_x is not None and audio_lengths is not None:
                modality_feats["audio"] = self.audio_encoder(audio_x, audio_lengths)
            if hasattr(self, "video_encoder") and video_x is not None and video_lengths is not None:
                modality_feats["video"] = self.video_encoder(video_x, video_lengths)
            if lyrics_x is not None and lyrics_lengths is not None and hasattr(self, "lyrics_encoder"):
                modality_feats["lyrics"] = self.lyrics_encoder(lyrics_x, lyrics_lengths)
            if comments_x is not None and comments_lengths is not None and hasattr(self, "comments_encoder"):
                modality_feats["comments"] = self.comments_encoder(comments_x, comments_lengths)
            logits = self.fusion(modality_feats)
            return logits
        elif self.fusion_type in ("cross_attention", "cross_attention_mean"):
            modality_seqs = {}
            modality_lengths = {}
            if audio_x is not None and audio_lengths is not None:
                modality_seqs["audio"] = audio_x
                modality_lengths["audio"] = audio_lengths
            if video_x is not None and video_lengths is not None:
                modality_seqs["video"] = video_x
                modality_lengths["video"] = video_lengths
            if lyrics_x is not None and lyrics_lengths is not None:
                modality_seqs["lyrics"] = lyrics_x
                modality_lengths["lyrics"] = lyrics_lengths
            if comments_x is not None and comments_lengths is not None:
                modality_seqs["comments"] = comments_x
                modality_lengths["comments"] = comments_lengths
            logits = self.fusion(
                modality_seqs=modality_seqs,
                modality_lengths=modality_lengths,
                video_indices=video_indices,
            )
            return logits
        elif self.fusion_type == "video_query_cross_attention":
            if video_x is None or video_lengths is None:
                raise RuntimeError("video_query_cross_attention requires video inputs")
            logits = self.fusion(
                video_x,
                video_lengths,
                audio_x,
                audio_lengths,
                lyrics_x,
                lyrics_lengths,
                comments_x,
                comments_lengths,
                video_indices,
            )
            return logits
        return torch.zeros(
            video_x.size(0),
            self.classifier.out_features if hasattr(self, "classifier") else 1,
            device=video_x.device,
        )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: List[str],
    use_audio: bool = True,
    use_video: bool = True,
    use_lyrics: bool = True,
    use_comments: bool = True,
) -> tuple:
    """Evaluate model on a dataset."""
    model.eval()
    preds = []
    trues = []
    keys = []

    with torch.no_grad():
        for batch in loader:
            (
                video_paths,
                audio_paths,
                lyrics_paths,
                comments_paths,
                video_x,
                audio_x,
                lyrics_x,
                comments_x,
                video_lengths,
                audio_lengths,
                lyrics_lengths,
                comments_lengths,
                y,
                video_indices,
            ) = batch

            for v, a, ly, co in zip(video_paths, audio_paths, lyrics_paths, comments_paths):
                key = v if v is not None else a if a is not None else ly if ly is not None else co
                keys.append(str(key.stem) if key is not None else None)

            if use_video and video_x is not None and video_lengths is not None:
                video_x = video_x.to(device)
                video_lengths = video_lengths.to(device)
            else:
                video_x = None
                video_lengths = None
            if use_audio and audio_x is not None and audio_lengths is not None:
                audio_x = audio_x.to(device)
                audio_lengths = audio_lengths.to(device)
            else:
                audio_x = None
                audio_lengths = None
            if use_lyrics and lyrics_x is not None and lyrics_lengths is not None:
                lyrics_x = lyrics_x.to(device)
                lyrics_lengths = lyrics_lengths.to(device)
            else:
                lyrics_x = None
                lyrics_lengths = None
            if use_comments and comments_x is not None and comments_lengths is not None:
                comments_x = comments_x.to(device)
                comments_lengths = comments_lengths.to(device)
            else:
                comments_x = None
                comments_lengths = None

            logits = model(
                video_x,
                audio_x,
                lyrics_x,
                comments_x,
                video_lengths,
                audio_lengths,
                lyrics_lengths,
                comments_lengths,
                video_indices,
            )
            pred_idxs = logits.argmax(dim=-1).cpu().numpy().tolist()
            
            for pred_idx, true_idx in zip(pred_idxs, y.tolist()):
                preds.append(labels[int(pred_idx)])
                trues.append(labels[int(true_idx)])
    
    df = pd.DataFrame({"key": keys, "pred_label": preds, "true_label": trues})
    acc = (df["pred_label"] == df["true_label"]).mean()
    return acc, df


def compute_f1(df: pd.DataFrame, labels: List[str]) -> dict:
    """Compute per-class and macro F1 scores."""
    per = {}
    f1s = []
    for lab in labels:
        tp = int(((df["pred_label"] == lab) & (df["true_label"] == lab)).sum())
        fp = int(((df["pred_label"] == lab) & (df["true_label"] != lab)).sum())
        fn = int(((df["pred_label"] != lab) & (df["true_label"] == lab)).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per[lab] = (prec, rec, f1)
        f1s.append(f1)
        
    macro_f1 = float(sum(f1s) / len(labels)) if len(labels) > 0 else 0.0
    return {"per_class": per, "macro_f1": macro_f1}


UNIFIED_SPLIT_FILENAME = "{split}_features.csv"


def _find_split_csv_in_root(root_dir: str, split: str, modality: str) -> Optional[str]:
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"{modality.capitalize()} root directory {root_dir} does not exist")
    if not root.is_dir():
        raise ValueError(f"{modality.capitalize()} root {root_dir} is not a directory")

    candidate = root / UNIFIED_SPLIT_FILENAME.format(split=split)
    if candidate.is_file():
        return str(candidate)
    return None


def _resolve_split_csv(
    explicit_path: Optional[str],
    root_dir: Optional[str],
    split: str,
    modality: str,
    required: bool = False,
) -> Optional[str]:
    if explicit_path:
        return explicit_path
    if not root_dir:
        if required:
            raise ValueError(
                f"Missing {split} {modality} CSV: pass --{split}-{modality}-csv or --{modality}-root-dir"
            )
        return None
    resolved = _find_split_csv_in_root(root_dir, split, modality)
    if resolved:
        return resolved
    raise ValueError(
        f"Missing standard {split} {modality} CSV ({UNIFIED_SPLIT_FILENAME.format(split=split)}) under {root_dir}"
    )


def _path_for_modality(args, modality: str) -> Optional[str]:
    """Path (root_dir or train CSV) for inferring model name. Prefer root_dir."""
    root = getattr(args, f"{modality}_root_dir", None)
    if root:
        return root
    return getattr(args, f"train_{modality}_csv", None)


def _infer_audio_model_name(args) -> str:
    path = _path_for_modality(args, "audio")
    if not path:
        return "audio"
    p = path.lower()
    if "mert" in p:
        return "MERT"
    if "af3" in p:
        return "AF3"
    return "audio"


def _infer_video_model_name(args) -> str:
    path = _path_for_modality(args, "video")
    if not path:
        return "enet"
    p = path.lower()
    if "moede" in p:
        return "moede"
    if "mean" in p:
        return "enet_mean"
    return "enet"


def _build_setting_id(args) -> str:
    """Build setting ID for wandb group: model names + fusion_type + modality suffix.
    Single-modality fusion: no modality suffix. Cross-attn series: max 2 modalities.
    """
    ft = args.fusion_type
    use_audio = getattr(args, "use_audio", True)
    use_video = getattr(args, "use_video", True)
    use_lyrics = getattr(args, "use_lyrics", True)
    use_comments = getattr(args, "use_comments", True)
    modality_flags = (
        ("audio", use_audio),
        ("video", use_video),
        ("lyrics", use_lyrics),
        ("comments", use_comments),
    )
    active = sorted([name for name, flag in modality_flags if flag])

    if ft in ("audio_only", "video_only", "lyrics_only", "comments_only"):
        if ft == "audio_only":
            return f"{_infer_audio_model_name(args)}_audio_only"
        if ft == "video_only":
            return f"{_infer_video_model_name(args)}_video_only"
        return ft  # lyrics_only, comments_only

    if ft == "video_query_cross_attention":
        # Only video + one text: video+lyrics or video+comments (Q=video, K/V=text)
        mod_suffix = "+".join(active)
        return f"{_infer_video_model_name(args)}_video_query_cross_attention_{mod_suffix}"

    if ft in ("cross_attention", "cross_attention_mean"):
        # Max 2 modalities + Q/K/V roles so different configs are distinguishable
        mod_suffix = "+".join(active[:2]) if len(active) >= 2 else "+".join(active)
        audio_m = _infer_audio_model_name(args) if use_audio else None
        video_m = _infer_video_model_name(args) if use_video else None
        parts = [p for p in (audio_m, video_m) if p and p not in ("audio", "video")]
        q = getattr(args, "cross_attn_q", "video")
        k = getattr(args, "cross_attn_k", "audio")
        v = getattr(args, "cross_attn_v", "audio")
        qkv = f"q{q}_k{k}_v{v}"
        return f"{'_'.join(parts)}_{ft}_{mod_suffix}_{qkv}"

    # late_concat or others: all active modalities
    mod_suffix = "+".join(active) if active else "none"
    audio_m = _infer_audio_model_name(args) if use_audio else None
    video_m = _infer_video_model_name(args) if use_video else None
    parts = [p for p in (audio_m, video_m) if p and p not in ("audio", "video")]
    return f"{'_'.join(parts)}_{ft}_{mod_suffix}"


def _resolve_modality_csv(args, split: str, modality: str, required: bool) -> Optional[str]:
    attr = f"{split}_{modality}_csv"
    explicit = getattr(args, attr)
    root_dir = getattr(args, f"{modality}_root_dir")
    return _resolve_split_csv(explicit, root_dir, split, modality, required)


def infer_feature_dims(dataset: EmoMVMultimodalDataset) -> tuple:
    """Infer feature dimensions. Uses first sample with each modality present so lyrics/comments show (seq_len, D) not placeholder (1, D)."""
    video_sample = audio_sample = lyrics_sample = comments_sample = None
    for idx in range(len(dataset)):
        s = dataset[idx]
        v_path, a_path, l_path, c_path = s[0], s[1], s[2], s[3]
        if video_sample is None and v_path is not None:
            video_sample = s[4]
        if audio_sample is None and a_path is not None:
            audio_sample = s[5]
        if lyrics_sample is None and l_path is not None:
            lyrics_sample = s[6]
        if comments_sample is None and c_path is not None:
            comments_sample = s[7]
        if video_sample is not None and audio_sample is not None:
            if (not dataset.has_lyrics or lyrics_sample is not None) and (not dataset.has_comments or comments_sample is not None):
                break
    if video_sample is None:
        video_sample = dataset[0][4]
    if audio_sample is None:
        audio_sample = dataset[0][5]
    if lyrics_sample is None and dataset.has_lyrics:
        lyrics_sample = dataset[0][6]
    if comments_sample is None and dataset.has_comments:
        comments_sample = dataset[0][7]
    print("Sample video feature shape:", video_sample.shape)
    print("Sample audio feature shape:", audio_sample.shape)
    if lyrics_sample is not None:
        note = " (placeholder: file missing?)" if lyrics_sample.shape[0] == 1 else ""
        print("Sample lyrics feature shape:", lyrics_sample.shape, note)
    if comments_sample is not None:
        note = " (placeholder: file missing?)" if comments_sample.shape[0] == 1 else ""
        print("Sample comments feature shape:", comments_sample.shape, note)

    if video_sample.dim() == 2:
        video_dim = video_sample.shape[1]
        number_of_experts = 1
    elif video_sample.dim() == 3:
        video_dim = video_sample.shape[2]
        number_of_experts = video_sample.shape[1]
    else:
        raise RuntimeError(f"Unexpected video feature shape: {video_sample.shape}")

    if audio_sample.dim() == 3:
        audio_dim = audio_sample.shape[2]
    elif audio_sample.dim() == 2:
        audio_dim = audio_sample.shape[1]
    else:
        raise RuntimeError(f"Unexpected audio feature shape: {audio_sample.shape}")

    lyrics_dim = 0
    if lyrics_sample is not None:
        if lyrics_sample.dim() == 2:
            lyrics_dim = lyrics_sample.shape[1]
        elif lyrics_sample.dim() == 1:
            lyrics_dim = lyrics_sample.shape[0]
        else:
            raise RuntimeError(f"Unexpected lyrics feature shape: {lyrics_sample.shape}")

    comments_dim = 0
    if comments_sample is not None:
        if comments_sample.dim() == 2:
            comments_dim = comments_sample.shape[1]
        elif comments_sample.dim() == 1:
            comments_dim = comments_sample.shape[0]
        else:
            raise RuntimeError(f"Unexpected comments feature shape: {comments_sample.shape}")

    return audio_dim, video_dim, lyrics_dim, comments_dim, number_of_experts


def build_model_config(args) -> tuple:
    """Build audio, video, lyrics, comments, and fusion configs from args."""
    audio_config = {
        "model_type": args.audio_model_type,
        "gru_hidden": args.audio_hidden,
        "gru_layers": args.audio_layers,
        "bidirectional": args.audio_bidirectional,
        "gru_dropout": args.audio_dropout,
        "tr_nhead": args.audio_tr_nhead,
        "tr_layers": args.audio_layers,
        "tr_dropout": args.audio_dropout,
        "attn_dropout": args.audio_dropout,
        "linear_dim": args.audio_hidden,
        "activation": args.audio_activation,
    }
    video_config = {
        "lstm_hidden": args.video_hidden,
        "dropout": args.video_dropout,
        "num_experts": args.video_num_experts,
    }
    text_like_config = {
        "model_type": args.text_model_type,
        "gru_hidden": args.text_hidden,
        "gru_layers": args.text_layers,
        "bidirectional": args.text_bidirectional,
        "gru_dropout": args.text_dropout,
        "tr_nhead": args.text_tr_nhead,
        "tr_layers": args.text_layers,
        "tr_dropout": args.text_dropout,
        "attn_dropout": args.text_dropout,
        "linear_dim": args.text_hidden,
        "activation": args.text_activation,
    }
    lyrics_config = dict(text_like_config)
    comments_config = dict(text_like_config)
    fusion_config = {
        "dropout": args.fusion_dropout,
        "hidden_dim": args.fusion_hidden,
        "cross_attn_heads": args.cross_attn_heads,
    }
    return audio_config, video_config, lyrics_config, comments_config, fusion_config


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal (Audio+Video) Emotion Recognition for EmoMV"
    )
    
    # Mode and data
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="train: train and validate; test: evaluate on test set",
    )
    parser.add_argument(
        "--no-audio",
        dest="use_audio",
        action="store_false",
        help="Disable audio modality (still load audio CSV but ignore during fusion)",
    )
    parser.add_argument(
        "--no-video",
        dest="use_video",
        action="store_false",
        help="Disable video modality",
    )
    parser.add_argument(
        "--no-lyrics",
        dest="use_lyrics",
        action="store_false",
        help="Disable lyrics modality",
    )
    parser.add_argument(
        "--no-comments",
        dest="use_comments",
        action="store_false",
        help="Disable comments modality",
    )
    parser.add_argument(
        "--allow-missing",
        dest="allow_missing_modalities",
        action="store_true",
        help="Keep samples when a modality is absent; fill missing with zero vectors.",
    )
    # Data: Separate modality CSVs per split
    parser.add_argument("--train-video-csv", help="Train video CSV (or use --video-root-dir)")
    parser.add_argument("--train-audio-csv", help="Train audio CSV (or use --audio-root-dir)")
    parser.add_argument("--train-lyrics-csv", help="Train lyrics CSV (or use --lyrics-root-dir)")
    parser.add_argument("--train-comments-csv", help="Train comments CSV (or use --comments-root-dir)")
    parser.add_argument("--val-video-csv", help="Validation video CSV")
    parser.add_argument("--val-audio-csv", help="Validation audio CSV")
    parser.add_argument("--val-lyrics-csv", help="Validation lyrics CSV")
    parser.add_argument("--val-comments-csv", help="Validation comments CSV")
    parser.add_argument("--test-video-csv", help="Test video CSV")
    parser.add_argument("--test-audio-csv", help="Test audio CSV")
    parser.add_argument("--test-lyrics-csv", help="Test lyrics CSV")
    parser.add_argument("--test-comments-csv", help="Test comments CSV")
    parser.add_argument("--audio-root-dir", help="Root directory containing train/val/test audio manifests")
    parser.add_argument("--video-root-dir", help="Root directory containing train/val/test video manifests")
    parser.add_argument("--lyrics-root-dir", help="Root directory containing train/val/test lyrics manifests (e.g. TER/csvs)")
    parser.add_argument("--comments-root-dir", help="Root directory containing train/val/test comments manifests (e.g. TER/comment_features)")
    parser.add_argument(
        "--output-base", required=True, help="Directory to write outputs"
    )
    
    # Model architecture
    parser.add_argument(
        "--fusion-type",
        choices=[
            "late_concat",
            "audio_only",
            "video_only",
            "lyrics_only",
            "comments_only",
            "cross_attention",
            "cross_attention_mean",
            "video_query_cross_attention",
        ],
        default="late_concat",
        help="Fusion strategy.",
    )
    parser.add_argument("--audio-model-type", default="mean_pool", help="Audio encoder model type")
    parser.add_argument("--audio-hidden", type=int, default=64, help="Audio encoder hidden size")
    parser.add_argument("--audio-layers", type=int, default=1, help="Audio encoder number of layers")
    parser.add_argument("--audio-bidirectional", action="store_true")
    parser.add_argument("--audio-tr-nhead", type=int, default=4)
    parser.add_argument("--audio-activation", default="relu")
    parser.add_argument("--audio-dropout", type=float, default=0.1)

    parser.add_argument("--text-model-type", default="mean_pool", help="Text encoder model type")
    parser.add_argument("--text-hidden", type=int, default=64, help="Text encoder hidden size")
    parser.add_argument("--text-layers", type=int, default=1, help="Text encoder number of layers")
    parser.add_argument("--text-bidirectional", action="store_true")
    parser.add_argument("--text-tr-nhead", type=int, default=4)
    parser.add_argument("--text-activation", default="relu")
    parser.add_argument("--text-dropout", type=float, default=0.1)
    
    parser.add_argument("--video-hidden", type=int, default=256)
    parser.add_argument("--video-num-experts", type=int, default=1)
    parser.add_argument("--video-dropout", type=float, default=0.1)
    
    parser.add_argument("--fusion-hidden", type=int, default=128)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--cross-attn-heads", type=int, default=4)
    parser.add_argument("--cross-attn-q", type=str, default="video", choices=["audio", "video", "lyrics", "comments"])
    parser.add_argument("--cross-attn-k", type=str, default="audio", choices=["audio", "video", "lyrics", "comments"])
    parser.add_argument("--cross-attn-v", type=str, default="audio", choices=["audio", "video", "lyrics", "comments"])
    parser.add_argument("--no-cross-attn-bidirectional", action="store_true", help="Disable bidirectional cross-attention")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model loading/saving
    parser.add_argument("--model-path", help="Path to existing model for testing")
    parser.add_argument("--save-model", default="best_model.pt", help="Filename for best model")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="EmoMV_Multimodal")
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Append one row per run (setting_id, lr, best_val_acc, best_val_f1, test_acc, test_f1, run_id) for paper tables. Default: none.",
    )

    args = parser.parse_args()
    args.use_audio = getattr(args, "use_audio", True)
    args.use_video = getattr(args, "use_video", True)
    args.use_lyrics = getattr(args, "use_lyrics", True)
    args.use_comments = getattr(args, "use_comments", True)

    out_dir = Path(args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        "Active modalities:",
        f"audio={args.use_audio}",
        f"video={args.use_video}",
        f"lyrics={args.use_lyrics}",
        f"comments={args.use_comments}",
    )

    labels = EMOMV_CLASS_LABELS
    num_classes = len(labels)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    test_acc = None
    test_f1 = None

    wandb_run = None
    modality_flags = (
        ("audio", args.use_audio),
        ("video", args.use_video),
        ("lyrics", args.use_lyrics),
        ("comments", args.use_comments),
    )
    active_mods = [name for name, flag in modality_flags if flag]
    modalities_suffix = "+".join(active_mods) if active_mods else "none"
    setting_id = _build_setting_id(args)
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                group=setting_id,
                name=f"lr{args.lr}",
            )
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
    
    # --- TRAINING MODE ---
    if args.mode == "train":
        train_video_csv = _resolve_modality_csv(args, "train", "video", required=True)
        train_audio_csv = _resolve_modality_csv(args, "train", "audio", required=True)
        train_lyrics_csv = _resolve_modality_csv(args, "train", "lyrics", required=args.use_lyrics)
        train_comments_csv = _resolve_modality_csv(args, "train", "comments", required=args.use_comments)
        val_video_csv = _resolve_modality_csv(args, "val", "video", required=True)
        val_audio_csv = _resolve_modality_csv(args, "val", "audio", required=True)
        val_lyrics_csv = _resolve_modality_csv(args, "val", "lyrics", required=args.use_lyrics)
        val_comments_csv = _resolve_modality_csv(args, "val", "comments", required=args.use_comments)

        train_set = EmoMVMultimodalDataset(
            train_video_csv,
            train_audio_csv,
            lyrics_csv=train_lyrics_csv,
            comments_csv=train_comments_csv,
            labels=labels,
            allow_missing_modalities=args.allow_missing_modalities,
        )
        val_set = EmoMVMultimodalDataset(
            val_video_csv,
            val_audio_csv,
            lyrics_csv=val_lyrics_csv,
            comments_csv=val_comments_csv,
            labels=labels,
            allow_missing_modalities=args.allow_missing_modalities,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_multimodal,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_multimodal,
        )

        audio_dim, video_dim, lyrics_dim, comments_dim, number_of_experts = infer_feature_dims(train_set)
        print(
            f"Audio dim: {audio_dim}, Video dim: {video_dim}, Lyrics dim: {lyrics_dim}, Comments dim: {comments_dim}, Experts: {number_of_experts}"
        )
        if number_of_experts != args.video_num_experts:
            print(
                f"Warning: Inferred number_of_experts ({number_of_experts}) does not match args.video_num_experts ({args.video_num_experts}). Using inferred value."
            )
            args.video_num_experts = number_of_experts
        audio_config, video_config, lyrics_config, comments_config, fusion_config = build_model_config(args)

        effective_audio_dim = audio_dim if args.use_audio else 0
        effective_video_dim = video_dim if args.use_video else 0
        effective_lyrics_dim = lyrics_dim if args.use_lyrics else 0
        effective_comments_dim = comments_dim if args.use_comments else 0

        model = MultimodalEmotionClassifier(
            audio_dim=effective_audio_dim,
            video_dim=effective_video_dim,
            lyrics_dim=effective_lyrics_dim,
            comments_dim=effective_comments_dim,
            audio_config=audio_config,
            video_config=video_config,
            lyrics_config=lyrics_config,
            comments_config=comments_config,
            fusion_type=args.fusion_type,
            fusion_config=fusion_config,
            num_classes=num_classes,
            cross_attn_q=args.cross_attn_q,
            cross_attn_k=args.cross_attn_k,
            cross_attn_v=args.cross_attn_v,
            cross_attn_bidirectional=not args.no_cross_attn_bidirectional,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        save_path = out_dir / args.save_model
        no_improve = 0
        
        for epoch in range(1, args.epochs + 1):
            # Training Loop
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Train E{epoch}", leave=False):
                (
                    _,
                    _,
                    _,
                    _,
                    video_x,
                    audio_x,
                    lyrics_x,
                    comments_x,
                    video_lengths,
                    audio_lengths,
                    lyrics_lengths,
                    comments_lengths,
                    y,
                    video_indices,
                ) = batch

                if args.use_video and video_x is not None and video_lengths is not None:
                    video_x = video_x.to(device)
                    video_lengths = video_lengths.to(device)
                else:
                    video_x = None
                    video_lengths = None
                if args.use_audio and audio_x is not None and audio_lengths is not None:
                    audio_x = audio_x.to(device)
                    audio_lengths = audio_lengths.to(device)
                else:
                    audio_x = None
                    audio_lengths = None
                if args.use_lyrics and lyrics_x is not None and lyrics_lengths is not None:
                    lyrics_x = lyrics_x.to(device)
                    lyrics_lengths = lyrics_lengths.to(device)
                else:
                    lyrics_x = None
                    lyrics_lengths = None
                if args.use_comments and comments_x is not None and comments_lengths is not None:
                    comments_x = comments_x.to(device)
                    comments_lengths = comments_lengths.to(device)
                else:
                    comments_x = None
                    comments_lengths = None
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(
                    video_x,
                    audio_x,
                    lyrics_x,
                    comments_x,
                    video_lengths,
                    audio_lengths,
                    lyrics_lengths,
                    comments_lengths,
                    video_indices,
                )
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            train_loss = total_loss / total
            train_acc = correct / total
            
            # Validation Loop
            val_acc, val_df = evaluate(
                model,
                val_loader,
                device,
                labels,
                use_audio=args.use_audio,
                use_video=args.use_video,
                use_lyrics=args.use_lyrics,
                use_comments=args.use_comments,
            )
            val_metrics = compute_f1(val_df, labels)
            val_f1 = val_metrics["macro_f1"]
            
            print(
                f"Epoch {epoch}: Loss={train_loss:.4f} TrAcc={train_acc:.4f} "
                f"ValAcc={val_acc:.4f} ValF1={val_f1:.4f}"
            )
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_acc": val_acc, "val_f1": val_f1,
                })
            
            # Save best model (record best checkpoint's val acc and val f1)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), str(save_path))
                print(f"  * Saved new best model to {save_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    
    # --- TESTING MODE ---
    if args.mode == "test" or args.mode == "train":
        print("\nStarting Test Evaluation...")
        test_video_csv = _resolve_modality_csv(args, "test", "video", required=True)
        test_audio_csv = _resolve_modality_csv(args, "test", "audio", required=True)
        test_lyrics_csv = _resolve_modality_csv(args, "test", "lyrics", required=args.use_lyrics)
        test_comments_csv = _resolve_modality_csv(args, "test", "comments", required=args.use_comments)
        test_set = EmoMVMultimodalDataset(
            test_video_csv,
            test_audio_csv,
            lyrics_csv=test_lyrics_csv,
            comments_csv=test_comments_csv,
            labels=labels,
            allow_missing_modalities=args.allow_missing_modalities,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_multimodal,
        )

        audio_dim, video_dim, lyrics_dim, comments_dim, number_of_experts = infer_feature_dims(test_set)
        print(
            f"Audio dim: {audio_dim}, Video dim: {video_dim}, Lyrics dim: {lyrics_dim}, Comments dim: {comments_dim}, Experts: {number_of_experts}"
        )
        if number_of_experts != args.video_num_experts:
            print(
                f"Warning: Inferred number_of_experts ({number_of_experts}) != args.video_num_experts ({args.video_num_experts}). Using inferred."
            )
            args.video_num_experts = number_of_experts

        audio_config, video_config, lyrics_config, comments_config, fusion_config = build_model_config(args)
        effective_audio_dim = audio_dim if args.use_audio else 0
        effective_video_dim = video_dim if args.use_video else 0
        effective_lyrics_dim = lyrics_dim if args.use_lyrics else 0
        effective_comments_dim = comments_dim if args.use_comments else 0
        model = MultimodalEmotionClassifier(
            audio_dim=effective_audio_dim,
            video_dim=effective_video_dim,
            lyrics_dim=effective_lyrics_dim,
            comments_dim=effective_comments_dim,
            audio_config=audio_config,
            video_config=video_config,
            lyrics_config=lyrics_config,
            comments_config=comments_config,
            fusion_type=args.fusion_type,
            fusion_config=fusion_config,
            num_classes=num_classes,
            cross_attn_q=args.cross_attn_q,
            cross_attn_k=args.cross_attn_k,
            cross_attn_v=args.cross_attn_v,
            cross_attn_bidirectional=not args.no_cross_attn_bidirectional,
        ).to(device)

        # 3. Load State Dict
        model_path = args.model_path or str(out_dir / args.save_model)
        if not Path(model_path).exists():
            print(f"Error: Model file {model_path} not found.")
        else:
            print(f"Loading weights from: {model_path}")
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
                print("Ensure test arguments (hidden dims, layers) match training arguments.")
                return

            # 4. Evaluate
            test_acc, test_df = evaluate(
                model,
                test_loader,
                device,
                labels,
                use_audio=args.use_audio,
                use_video=args.use_video,
                use_lyrics=args.use_lyrics,
                use_comments=args.use_comments,
            )
            test_metrics = compute_f1(test_df, labels)
            test_f1 = test_metrics["macro_f1"]
            
            preds_csv = out_dir / "test_predictions.csv"
            test_df.to_csv(preds_csv, index=False)
            
            print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            print(f"Predictions saved to {preds_csv}")
            
            if wandb_run:
                wandb_run.log({"test_acc": test_acc, "test_f1": test_f1})

    run_id = ""
    if wandb_run:
        run_id = getattr(wandb_run, "id", "") or ""
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["best_val_f1"] = best_val_f1
        wandb_run.summary["best_epoch"] = best_epoch
        if test_acc is not None:
            wandb_run.summary["test_acc"] = test_acc
        if test_f1 is not None:
            wandb_run.summary["test_f1"] = test_f1
        wandb_run.finish()

    # Local summary CSV for paper tables: one row per run when test was computed
    if getattr(args, "summary_csv", None) and test_acc is not None:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        ft = getattr(args, "fusion_type", "")
        is_cross = ft in ("cross_attention", "cross_attention_mean", "video_query_cross_attention")
        row = {
            "setting_id": setting_id,
            "lr": args.lr,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "cross_attn_q": getattr(args, "cross_attn_q", "") if is_cross else "",
            "cross_attn_k": getattr(args, "cross_attn_k", "") if is_cross else "",
            "cross_attn_v": getattr(args, "cross_attn_v", "") if is_cross else "",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }
        file_exists = summary_path.exists()
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        print(f"Appended summary row to {summary_path}")


if __name__ == "__main__":
    main()