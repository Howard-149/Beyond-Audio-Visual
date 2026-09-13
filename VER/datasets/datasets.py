"""Dataset classes for VER pipelines (CREMA-D, EmoMV, and manifest-based formats).

Each dataset class implements the same interface:
  - __len__: return number of samples
  - __getitem__(idx): return (Path, feature_tensor, label_idx)
    where feature_tensor is shape (T, D) or (T, E, D)
    and label_idx is an integer index into CLASS_LABELS
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CREMAD_CLASS_LABELS = [
    "A",  # Angry
    "D",  # Disgust
    "F",  # Fear
    "H",  # Happy
    "N",  # Neutral
    "S",  # Sad
]
EMOMV_CLASS_LABELS = ['exciting','fear','tense','sad','relax']


def get_label_string(label_idx: int, dataset_type: str = 'crema') -> str:
    """Convert label index to human-readable string.
    
    Args:
        label_idx: integer index (0-5 for CREMA-D, 0-4 for EmoMV)
        dataset_type: 'crema' or 'emomv'
    
    Returns:
        Label string (e.g., 'A' for CREMA-D, 'exciting' for EmoMV)
    """
    if dataset_type == 'crema':
        return CREMAD_CLASS_LABELS[int(label_idx)]
    elif dataset_type == 'emomv':
        return EMOMV_CLASS_LABELS[int(label_idx)]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def get_label_idx(label_str: str, dataset_type: str = 'crema') -> int:
    """Convert label string to index.
    
    Args:
        label_str: label name or numeric string (e.g., 'A', 'exciting', '0')
        dataset_type: 'crema' or 'emomv'
    
    Returns:
        Integer index
    """
    if dataset_type == 'crema':
        labels = CREMAD_CLASS_LABELS
    elif dataset_type == 'emomv':
        labels = EMOMV_CLASS_LABELS
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Try numeric index first
    if label_str.strip().isdigit():
        idx = int(label_str.strip())
        if 0 <= idx < len(labels):
            return idx
        raise ValueError(f"Numeric label {idx} out of range for {dataset_type}")
    
    # Try string match
    label_norm = label_str.strip().lower() if dataset_type == 'emomv' else label_str.strip().upper()
    if label_norm in labels:
        return labels.index(label_norm)
    
    raise ValueError(f"Unknown label '{label_str}' for dataset_type '{dataset_type}'")


# Export for backward compatibility
# CLASS_LABELS = CREMAD_CLASS_LABELS

class CremaDFeaturesDataset(Dataset):
    """Dataset for CREMA-D: loads .npy feature vectors and maps them to labels using a CSV.

    CSV format (columns):
    - filename: feature file stem (e.g., '1001_DFA_ANG_XX')
    - facevote: label in {A, D, F, H, N, S}

    Features expected:
    - .pt files stored in feature_dir (schema v1 format) with stems matching 'filename' column
    - Embedding tensor shape (T, D) for simple features or (T, E, D) for multi-expert embeddings
    """

    def __init__(self, feature_dir: str, label_csv: str):
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.exists():
            raise RuntimeError(f"feature_dir not found: {feature_dir}")
        self.num_classes = len(CREMAD_CLASS_LABELS)
        df = pd.read_csv(label_csv, dtype=str)
        expected_key = "filename"
        expected_label = "facevote"
        if expected_key not in df.columns or expected_label not in df.columns:
            raise RuntimeError(
                f"CSV must contain columns '{expected_key}' and '{expected_label}'. Found: {df.columns.tolist()}"
            )

        key_series = df[expected_key].astype(str)
        label_series = df[expected_label].astype(str)

        canon_set = set(CREMAD_CLASS_LABELS)
        self.map = {}
        for k, lab in zip(key_series.tolist(), label_series.tolist()):
            if pd.isna(k) or pd.isna(lab):
                continue
            k_str = str(k)
            k_stem = Path(k_str).stem
            lab_str = str(lab).strip()
            lab_norm = lab_str.upper()
            if lab_norm not in canon_set:
                raise RuntimeError(
                    f'Found unsupported label "{lab_str}" for key {k_stem}; expected one of {CREMAD_CLASS_LABELS}'
                )
            if k_stem in self.map and self.map[k_stem] != lab_norm:
                raise RuntimeError(
                    f'Conflicting labels for key {k_stem} in CSV: "{self.map[k_stem]}" vs "{lab_norm}"'
                )
            self.map[k_stem] = lab_norm

        self.samples: List[Tuple[Path, str]] = []
        for p in sorted(self.feature_dir.glob("*.pt")):
            stem = p.stem
            if stem in self.map:
                lab = self.map[stem]
                self.samples.append((p, lab))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No matching .npy features found for provided CSV and key/label configuration"
            )

        self.labels = CREMAD_CLASS_LABELS
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, lab = self.samples[idx]
        if str(p).endswith('.pt'):
            data = torch.load(p, map_location="cpu")
            # Extract embedding from schema v1 format
            x = data["embedding"]
        else:
            raise ValueError(f"Unsupported feature format: {p}. Expected .pt file (schema v1).")
        
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (T, 1, D) -> (T, D)
        y = self.label2idx[lab]
        return p, x, y


class EmoMVFeaturesDataset(Dataset):
    """Dataset for manifest-based feature lists (e.g., EmoMV, social media predictions).

    CSV format (columns):
    - feature_path: absolute or relative path to a .npy file
    - video_label: ground-truth label as integer (0-4) or string ('exciting', 'fear', etc.)

    Optional columns:
    - indices: comma- or space-separated frame indices (used during inference only)

    Features expected:
    - .pt files (schema v1 format) at paths specified in feature_path column
    - Embedding tensor shape (T, D) or (T, E, D)
    """

    def __init__(self, label_csv: str):
        self.num_classes = len(EMOMV_CLASS_LABELS)
        df = pd.read_csv(label_csv, dtype=str)
        required = ["feature_path", "video_label"]
        for r in required:
            if r not in df.columns:
                raise RuntimeError(
                    f"Manifest must contain columns {required}. Found: {df.columns.tolist()}"
                )

        self.labels = EMOMV_CLASS_LABELS
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        # Also support numeric labels (0-4 -> exciting, fear, tense, sad, relax)
        for i, l in enumerate(self.labels):
            self.label2idx[str(i)] = i

        samples: List[Tuple[Path, str]] = []
        for _, row in df.iterrows():
            fp = Path(str(row["feature_path"]).strip())
            lab_str = str(row["video_label"]).strip()

            if not fp.exists():
                # Skip missing files silently to allow partial manifests
                continue

            # Normalize label: numeric or string (case-insensitive for strings)
            try:
                # Try numeric first
                if lab_str.isdigit():
                    lab_idx = int(lab_str)
                    if 0 <= lab_idx < len(self.labels):
                        lab_norm = str(lab_idx)  # keep as numeric string for consistency
                    else:
                        raise ValueError(f"Numeric label {lab_idx} out of range")
                else:
                    # String label (case-insensitive match)
                    lab_lower = lab_str.lower()
                    if lab_lower in self.labels:
                        lab_norm = lab_lower
                    else:
                        raise ValueError(f"Unknown label '{lab_str}'")
            except ValueError as e:
                raise RuntimeError(
                    f"Unsupported label '{lab_str}' in manifest; expected one of {EMOMV_CLASS_LABELS} or 0-4: {e}"
                )

            samples.append((fp, lab_norm))

        if len(samples) == 0:
            raise RuntimeError("No valid samples found in manifest")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, lab = self.samples[idx]
        if str(p).endswith('.pt'):
            data = torch.load(str(p), map_location="cpu")
            # Extract embedding from schema v1 format
            x = data["embedding"]
        else:
            raise ValueError(f"Unsupported feature format: {p}. Expected .pt file (schema v1).")
        
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (T, 1, D) -> (T, D)
        y = self.label2idx[lab]
        return p, x, y


def make_dataset(
    dataset_type: str,
    feature_dir: str = None,
    label_csv: str = None,
) -> Dataset:
    """Factory function to create dataset instances.

    Args:
        dataset_type: one of 'crema', 'emomv', or 'manifest'
        feature_dir: directory with .npy files (required for 'crema')
        label_csv: CSV file with labels (required for both)

    Returns:
        Dataset instance

    Raises:
        ValueError if dataset_type is unknown or required args are missing
    """
    if dataset_type == "crema":
        if feature_dir is None or label_csv is None:
            raise ValueError(
                "dataset_type 'crema' requires both --feature-dir and label CSV"
            )
        return CremaDFeaturesDataset(feature_dir, label_csv)
    elif dataset_type == "emomv":
        if label_csv is None:
            raise ValueError(f"dataset_type '{dataset_type}' requires a label CSV")
        return EmoMVFeaturesDataset(label_csv)
    else:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'; expected 'crema', 'emomv', or 'manifest'"
        )
