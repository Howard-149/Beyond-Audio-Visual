# Embedding Storage Format v1.0

This document describes the unified embedding storage schema used across audio and video feature extraction pipelines.

## Overview

All embeddings are now stored as `.pt` files using `torch.save()` with a standardized schema defined in `utils/embedding_schema.py`. This replaces the previous mixed format (`.npy` for video, ad-hoc `.pt` for audio).

## Schema Definition

See `utils/embedding_schema.py` for the complete `EmbeddingV1` dataclass and validation logic.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `torch.FloatTensor` | The actual embedding tensor |
| `modality` | `str` | Source modality: `"audio"`, `"video"`, or `"text"` |
| `model_name` | `str` | Model identifier (e.g., `"m-a-p/MERT-v1-95M"`) |
| `layer` | `str` | Layer or representation source (e.g., `"last_hidden_state"`) |
| `pooling` | `str` | Pooling strategy: `"none"`, `"mean"`, `"max"`, `"attention"`, `"cls"` |
| `embedding_dim` | `int` | Feature dimension `D` |
| `temporal_dim` | `Optional[int]` | Temporal length `T` (None for single-vector embeddings) |
| `schema_version` | `str` | Must be `"1.0"` |
| `created_at` | `str` | ISO 8601 timestamp (UTC) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `sample_rate` | `Optional[int]` | Audio sampling rate in Hz |
| `fps` | `Optional[float]` | Video frames per second |
| `extra_metadata` | `Dict[str, Any]` | Free-form metadata (label, filename, paths, etc.) |

### Tensor Shape Conventions

- **Sequence embeddings**: `(T, D)` where `T = temporal_dim` and `D = embedding_dim`
- **Single-vector embeddings**: `(D,)` where `temporal_dim = None`
- **Multi-expert video**: `(T, E, D)` where `E` is number of experts
- Batch dimension is **not allowed** in stored embeddings

## File Format

### Audio Features (`.pt`)

Example from `MER/extract_EmoMV.py`:

```python
{
    "embedding": torch.FloatTensor(shape=(T, 768)),  # MERT last_hidden_state
    "modality": "audio",
    "model_name": "m-a-p/MERT-v1-95M",
    "layer": "last_hidden_state",
    "pooling": "none",
    "embedding_dim": 768,
    "temporal_dim": 120,
    "schema_version": "1.0",
    "created_at": "2025-12-14T10:30:00Z",
    "sample_rate": 24000,
    "fps": None,
    "extra_metadata": {
        "label": 0,
        "filename": "clip_0001",
        "source_rel": "EmoMV/train/clip_0001.mp4",
        "subset": "EmoMV",
        "split": "train"
    }
}
```

### Video Features (`.pt`)

Example from `VER/extract_EmoMV.py`:

```python
{
    "embedding": torch.FloatTensor(shape=(T, 1280)),  # EmotiEffNet features
    "modality": "video",
    "model_name": "enet_b0_8_best_afew",
    "layer": "feature_extraction",
    "pooling": "none",
    "embedding_dim": 1280,
    "temporal_dim": 150,
    "schema_version": "1.0",
    "created_at": "2025-12-14T10:32:00Z",
    "sample_rate": None,
    "fps": 30.0,
    "extra_metadata": {
        "clipName": "clip_0001",
        "indices": "0,10,20,30,...",
        "subset": "Dataset1",
        "split": "train",
        "source_path": "/path/to/video.mp4"
    }
}
```

## CSV Manifest Format

All extractors now produce standardized CSV manifests with columns:

```csv
key,feature_path,label
clip_0001,/path/to/features/clip_0001.pt,0
clip_0002,/path/to/features/clip_0002.pt,exciting
```

- `key`: Unique clip identifier (filename stem)
- `feature_path`: Absolute or relative path to `.pt` file
- `label`: String label or numeric index (0-4 for EmoMV)

## Dataset Loaders

All dataset classes support both legacy `.npy` and new `.pt` formats:

- `VER/datasets/datasets.py`: `CremaDFeaturesDataset`, `EmoMVFeaturesDataset`
- `data/multimodal.py`: `MultimodalEmoMVDataset`, `MultimodalFromCSVs`

Loaders automatically detect file extension and load accordingly:

```python
if str(path).endswith('.npy'):
    arr = np.load(path)
    x = torch.tensor(arr, dtype=torch.float32)
elif str(path).endswith('.pt'):
    data = torch.load(path)
    x = data["embedding"]
```

## Migration Guide

### Extracting New Features

Audio:
```bash
python MER/extract_EmoMV.py /path/to/EmoMV_train.csv \
  --out_dir MER/EmoMV_audio/train \
  --data_root /path/to/EmoMV/media \
  --subset EmoMV --split train
```

Video:
```bash
python VER/extract_EmoMV.py \
  --input-base /path/to/EmoMV/media \
  --output-dir VER/EmoMV_features \
  --subsets Dataset1 \
  --model-name enet_b0_8_best_afew
```

Both scripts now:
1. Save embeddings as `.pt` with EmbeddingSchema v1
2. Generate standardized CSV manifests
3. Validate schema before saving

### Using Legacy Features

Existing `.npy` files continue to work without modification. Dataset loaders transparently handle both formats.

### Validating Embeddings

```python
from utils.embedding_schema import validate_embedding_schema_v1
import torch

data = torch.load("features/clip_0001.pt")
validate_embedding_schema_v1(data)  # Raises ValueError if invalid
```

## Benefits

- **Unified format**: Single storage format for all modalities
- **Rich metadata**: Embedded provenance, model info, and creation time
- **Type safety**: Explicit schema with validation
- **Extensibility**: `extra_metadata` and versioning support future needs
- **Backward compatible**: Loaders support legacy `.npy` files
- **PyTorch-native**: No NumPy/PyTorch conversion overhead

## References

- Schema definition: `utils/embedding_schema.py`
- Audio extraction: `MER/extract_EmoMV.py`
- Video extraction: `VER/extract_EmoMV.py`
- Dataset loaders: `VER/datasets/datasets.py`, `data/multimodal.py`
- Multimodal training: `multimodal_classifier.py`
