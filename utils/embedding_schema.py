"""EmbeddingSchema v1: Extensible multimodal embedding container for PyTorch.

This module defines a formal, versioned schema for storing embeddings to disk
as `.pt` files via `torch.save`. It is designed for long-term use in research and
production training pipelines, supporting audio, video, and future modalities
such as text. The schema is explicit, PyTorch-centric, and backward-compatible.

Usage:
- Create an `EmbeddingV1` instance or a plain dict matching the schema.
- Save to disk with `torch.save(instance.__dict__, path)` or `torch.save(dict_obj, path)`.
- Load with `torch.load(path)` and validate via `validate_embedding_schema_v1`.

Non-goals: Dataset or dataloader logic is intentionally excluded.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

import torch

SchemaVersion = Literal["1.0"]


@dataclass
class EmbeddingV1:
    """Formal schema for multimodal embeddings (version 1.0).

    Required fields:
    - embedding: torch.FloatTensor
    The actual embedding tensor. For sequences, shape is (T, D) or multi-expert (T, E, D).
    For single-vector embeddings, shape is (D,). Batch dimension is not allowed.
    - modality: str
      The source modality. Examples: "audio", "video", "text".
    - model_name: str
      Name or identifier of the model used to compute the embedding (e.g., "m-a-p/MERT-v1-95M").
    - layer: str
      The model layer or representation source (e.g., "last_hidden_state", "layer_12").
    - pooling: str
      Pooling strategy used to convert sequences to a single vector; use "none" for sequence
      embeddings. Examples: "mean", "max", "attention", "cls", "none".
    - embedding_dim: int
      The feature dimension D of the embedding tensor.
    - temporal_dim: Optional[int]
      The temporal length T if the embedding is a sequence; None for single-vector embeddings.
    - schema_version: SchemaVersion
      Explicit version tag. Must be "1.0" for this schema.
    - created_at: str
      ISO 8601 timestamp of when the embedding was created.

    Optional fields (modality-specific and general):
    - sample_rate: Optional[int]
      Audio-only sampling rate in Hertz when relevant.
    - fps: Optional[float]
      Video-only frames per second when relevant.
    - extra_metadata: Dict[str, Any]
      Free-form dictionary for future extension (dataset info, source paths, etc.).
    """

    embedding: torch.FloatTensor
    modality: str
    model_name: str
    layer: str
    pooling: str
    embedding_dim: int
    temporal_dim: Optional[int]
    schema_version: SchemaVersion
    created_at: str

    sample_rate: Optional[int] = None
    fps: Optional[float] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation suitable for torch.save."""
        return {
            "embedding": self.embedding,
            "modality": self.modality,
            "model_name": self.model_name,
            "layer": self.layer,
            "pooling": self.pooling,
            "embedding_dim": self.embedding_dim,
            "temporal_dim": self.temporal_dim,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "sample_rate": self.sample_rate,
            "fps": self.fps,
            "extra_metadata": self.extra_metadata,
        }


def now_iso8601() -> str:
    """Return current time as ISO 8601 string (UTC)."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def validate_embedding_schema_v1(obj: Dict[str, Any]) -> None:
    """Validate that `obj` conforms to EmbeddingSchema v1.

    Raises descriptive `ValueError` if constraints are violated.

    Mandatory keys:
    - embedding (torch.FloatTensor)
    - modality (str)
    - model_name (str)
    - layer (str)
    - pooling (str)
    - embedding_dim (int)
    - temporal_dim (Optional[int])
    - schema_version (Literal["1.0"]) 
    - created_at (ISO 8601 string)

    Optional keys:
    - sample_rate (int, audio only)
    - fps (float, video only)
    - extra_metadata (dict)

        Shape semantics:
        - If temporal_dim is None, `embedding` MUST be 1-D of shape (D,).
        - If temporal_dim is not None, `embedding` MUST be either:
                - 2-D of shape (T, D), or
                - 3-D of shape (T, E, D) for multi-expert features.
            In all cases, T MUST equal temporal_dim and D MUST equal embedding_dim.
        - Batch dimension is not allowed.
    """
    required = [
        "embedding",
        "modality",
        "model_name",
        "layer",
        "pooling",
        "embedding_dim",
        "temporal_dim",
        "schema_version",
        "created_at",
    ]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing required field: {k}")

    # Types
    emb = obj["embedding"]
    if not isinstance(emb, torch.Tensor):
        raise ValueError("embedding must be a torch.Tensor")
    if emb.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("embedding must be a floating-point tensor")

    modality = obj["modality"]
    if not isinstance(modality, str) or len(modality) == 0:
        raise ValueError("modality must be a non-empty string")

    for sfield in ["model_name", "layer", "pooling", "created_at"]:
        if not isinstance(obj[sfield], str) or len(obj[sfield]) == 0:
            raise ValueError(f"{sfield} must be a non-empty string")

    if obj["schema_version"] != "1.0":
        raise ValueError("schema_version must be '1.0'")

    emb_dim = obj["embedding_dim"]
    if not isinstance(emb_dim, int) or emb_dim <= 0:
        raise ValueError("embedding_dim must be a positive int")

    temporal_dim = obj["temporal_dim"]
    if temporal_dim is not None and (not isinstance(temporal_dim, int) or temporal_dim <= 0):
        raise ValueError("temporal_dim must be None or a positive int")

    # Shape checks
    if temporal_dim is None:
        if emb.dim() != 1:
            raise ValueError("For single-vector embeddings, embedding must be 1-D (D,)")
        if emb.shape[0] != emb_dim:
            raise ValueError("embedding_dim does not match tensor shape")
    else:
        # Accept either (T, D) or (T, E, D)
        if emb.dim() == 2:
            T, D = emb.shape
        elif emb.dim() == 3:
            T, E, D = emb.shape
            if E <= 0:
                raise ValueError("expert dimension must be positive for 3-D embeddings (T, E, D)")
        else:
            raise ValueError("For sequence embeddings, embedding must be 2-D (T, D) or 3-D (T, E, D)")

        if D != emb_dim:
            raise ValueError("embedding_dim does not match last tensor dimension")
        if T != temporal_dim:
            raise ValueError("temporal_dim does not match first tensor dimension")

    # Optional fields
    if "sample_rate" in obj and obj["sample_rate"] is not None:
        sr = obj["sample_rate"]
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("sample_rate must be a positive int when provided")

    if "fps" in obj and obj["fps"] is not None:
        fps = obj["fps"]
        if not (isinstance(fps, (int, float)) and fps > 0):
            raise ValueError("fps must be a positive number when provided")

    if "extra_metadata" in obj and obj["extra_metadata"] is not None:
        if not isinstance(obj["extra_metadata"], dict):
            raise ValueError("extra_metadata must be a dict when provided")


# Examples

def example_audio_embedding() -> Dict[str, Any]:
    """Return a concrete example for an audio embedding (MERT, mean pooling).

    Example tensor shape: (T, D) with temporal_dim=T and embedding_dim=D.
    Pooling recorded as "mean" even if the stored tensor is per-frame; downstream
    consumers may also store the pooled vector separately as another instance.
    """
    T, D = 120, 768
    emb = torch.randn(T, D, dtype=torch.float32)
    obj = {
        "embedding": emb,
        "modality": "audio",
        "model_name": "m-a-p/MERT-v1-95M",
        "layer": "last_hidden_state",
        "pooling": "mean",
        "embedding_dim": D,
        "temporal_dim": T,
        "schema_version": "1.0",
        "created_at": now_iso8601(),
        "sample_rate": 24000,
        "fps": None,
        "extra_metadata": {
            "source_rel": "EmoMV/train/clip_0001.mp4",
            "subset": "EmoMV",
            "split": "train",
        },
    }
    validate_embedding_schema_v1(obj)
    return obj


def example_video_embedding() -> Dict[str, Any]:
    """Return a concrete example for a video embedding (VideoMAE, CLS token).

    Example tensor shape: (D,) for single-vector CLS token; temporal_dim=None.
    Pooling recorded as "cls".
    """
    D = 1024
    emb = torch.randn(D, dtype=torch.float32)
    obj = {
        "embedding": emb,
        "modality": "video",
        "model_name": "facebook/videomae-base",
        "layer": "cls_token",
        "pooling": "cls",
        "embedding_dim": D,
        "temporal_dim": None,
        "schema_version": "1.0",
        "created_at": now_iso8601(),
        "sample_rate": None,
        "fps": 30.0,
        "extra_metadata": {
            "source_rel": "EmoMV/train/clip_0001.mp4",
            "subset": "EmoMV",
            "split": "train",
        },
    }
    validate_embedding_schema_v1(obj)
    return obj


def example_text_embedding() -> Dict[str, Any]:
    """Return a concrete example for a text embedding (RoBERTa, mean pooling).

    Example tensor shape: (D,) for a pooled vector; temporal_dim=None.
    Pooling recorded as "mean".
    """
    D = 768
    emb = torch.randn(D, dtype=torch.float32)
    obj = {
        "embedding": emb,
        "modality": "text",
        "model_name": "cardiffnlp/twitter-roberta-base-2022-154m",
        "layer": "last_hidden_state",
        "pooling": "mean",
        "embedding_dim": D,
        "temporal_dim": None,
        "schema_version": "1.0",
        "created_at": now_iso8601(),
        "sample_rate": None,
        "fps": None,
        "extra_metadata": {
            "source_rel": "TER/csvs/text_train.csv",
            "subset": "EmoMV",
            "split": "train",
        },
    }
    validate_embedding_schema_v1(obj)
    return obj


__all__ = [
    "EmbeddingV1",
    "validate_embedding_schema_v1",
    "example_audio_embedding",
    "example_video_embedding",
    "example_text_embedding",
    "now_iso8601",
]